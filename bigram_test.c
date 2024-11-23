#include "ctorch.c"

typedef struct BigramArch {
    Queue *param_list;

    LinearLayer *ll1;
    ActivationLayer *sm;
    
    CrossEntropyLoss *loss;
} BigramArch;

void create_bigramarch(int in_feature_size, int out_feature_size, BigramArch **out_arch) {
    *out_arch = (BigramArch *)malloc(sizeof(BigramArch));

    // Initialize parameters' list
    (*out_arch)->param_list = create_queue();

    // Initialize DNN layers
    (*out_arch)->ll1 = linearlayer((*out_arch)->param_list, in_feature_size, out_feature_size, false);  // Input -> Softmax
    (*out_arch)->sm  = activation_layer(tensor_softmax, grad_tensor_softmax);                           // Softmax
    
    (*out_arch)->loss = crossentropyloss();                                                             // Cross Entropy Loss
}

void dispose_bigramarch(BigramArch *arch) {
    dispose_linearlayer(arch->ll1);
    dispose_activationlayer(arch->sm);

    dispose_crossentropyloss(arch->loss);

    dispose_queue(arch->param_list);

    free(arch);
}

void bigramforwad(BigramArch *arch, Node *n_X, bool inference_mode, Node **n_y_pred) {
    Node *n_y = NULL;

    forward_linearlayer(arch->ll1, n_X, &n_y);

    if (inference_mode) {
        forward_activationlayer_with_dim(arch->sm, n_y, 1, n_y_pred);
    } else {
        *n_y_pred = n_y;
    }
}

void bigramloss(BigramArch *arch, Node *n_y_pred, Node *n_target, Node **n_loss) {
    forward_crossentropyloss(arch->loss, n_y_pred, n_target, 1, n_loss);
}

// Function to load the Bigram dataset from names file
int load_bigram_dataset(
    const char *names_file,
    char **out_dataset_X,
    char **out_dataset_Y
) {
    FILE *file = fopen(names_file, "r");
    if (file == NULL) {
        printf("Error: Unable to open file %s\n", names_file);
        exit(1);
    }

    // Step 1: Initialize dynamic tmp array with a starting "."
    char *tmp = malloc(3);  // Start with 1 character for "." + 1 for null terminator
    if (tmp == NULL) {
        perror("Memory allocation error");
        fclose(file);
        exit(1);
    }
    tmp[0] = '.';
    tmp[1] = '.';
    tmp[2] = '\0';
    int current_idx = 1;
    int current_len = 3;

    // Step 2: Append each line with a leading "\n" (except the first line)
    char line[256] = {0};
    while (fgets(line, sizeof(line), file) != NULL) {
        int line_len = 0;
        while (line[line_len] >= 'a' && line[line_len] <= 'z') {
            line_len++;
        }

        // Calculate the new size needed for tmp and reallocate
        current_len += line_len + 1;
        tmp = realloc(tmp, current_len);
        if (tmp == NULL) {
            perror("Memory allocation error");
            fclose(file);
            exit(1);
        }

        int i = 0;
        while (line[i] >= 'a' && line[i] <= 'z') {
            tmp[current_idx++] = line[i++];
        }
        tmp[current_idx++] = '.';
    }
    tmp[current_idx] = '\0';
    fclose(file);

    // Create 2D array with moving window of 2 characters
    int tmp_len = strlen(tmp);
    *out_dataset_X = (char *)malloc((tmp_len - 1) * sizeof(char));
    *out_dataset_Y = (char *)malloc((tmp_len - 1) * sizeof(char));

    for (int i = 0; i < tmp_len - 1; i++) {
        (*out_dataset_X)[i] = tmp[i];
        (*out_dataset_Y)[i] = tmp[i + 1];
    }

    // Free allocated memory
    free(tmp);

    return tmp_len - 1;
}

void dispose_bigram_dataset(
    int dataset_size,
    char *dataset_X,
    char *dataset_Y
) {
    free(dataset_X);
    free(dataset_Y);
}

// Function to load a batch of Bigram data into tensors X and Y
void load_bigram_batch(
    char *dataset_X,
    char *dataset_Y,
    int dataset_size,
    int batch_idx,
    int batch_size,
    bool shuffle,
    Tensor **out_X,
    Tensor **out_Y
) {
    int start_idx = batch_idx * batch_size;
    int actual_batch_size = (start_idx + batch_size < dataset_size) ? batch_size : dataset_size - start_idx;

    create_tensor((int []){actual_batch_size, 27}, 2, out_X);   // From a-z plus "."
    create_tensor((int []){actual_batch_size, 27}, 2, out_Y);   // From a-z plus "."
    init_tensor(0.0, *out_X);
    init_tensor(0.0, *out_Y);

    // Loop through the batch and one_hot encode X/Y
    for (int i = 0; i < actual_batch_size; i++) {
        int src_idx = start_idx + i;

        char chr0 = dataset_X[src_idx];
        char chr1 = dataset_Y[src_idx];
        int enc_x = (chr0 == '.') ? 26 : chr0 - 'a';
        int enc_y = (chr1 == '.') ? 26 : chr1 - 'a';
        set_element(*out_X, 1.0, i, enc_x);
        set_element(*out_Y, 1.0, i, enc_y);
    }

    if (shuffle) {
        shuffle_data(*out_X, *out_Y);
    }
}

void bigram_train(
    const char *bigram_train_names_file
) {
    // Load the dataset
    char *dataset_X = NULL, *dataset_Y = NULL;
    int dataset_size = load_bigram_dataset(bigram_train_names_file, &dataset_X, &dataset_Y);
    int tokens_count = 27;

    // Define hyperparameters
    int training_size = dataset_size;
    int batch_size = training_size;
    int num_batches = ceil(training_size * 1.0 / batch_size);   // Number of batches in the epoch (adjust accordingly)
    int num_batches_to_print = 100;                             // Number of batches in the epoch to print result
    int epoch = 1000;
    double lr = 15;    // Learning rate

    Tensor *tensor_lr = NULL;
    create_tensor_from_scalar(lr, &tensor_lr);  // Learning rate as a scalar tensor

    // Create DNN layers
    BigramArch *arch = NULL;
    create_bigramarch(tokens_count, tokens_count, &arch);

    // load_bigram_batch loads the Bigram batch of tokens
    for (int e = 1; e <= epoch; e++) {
        printf("Epoch: %d/%d\n\n", e, epoch);
        
        double accumulated_epoch_loss = 0.0;
        for (int b = 0; b < num_batches; b++) {
            // Load a batch of data (X, Y)
            Tensor *X = NULL, *Y = NULL;
            load_bigram_batch(dataset_X, dataset_Y, training_size, b, batch_size, true, &X, &Y);

            // Convert tensors to nodes
            Node *n_X = NULL, *n_Y = NULL;
            create_leaf(X, false, &n_X);
            create_leaf(Y, false, &n_Y);

            // Forward pass through the DNN
            Node *n_y_pred = NULL;
            bigramforwad(arch, n_X, false, &n_y_pred);

            // Loss calculation
            Node *n_loss = NULL;
            bigramloss(arch, n_y_pred, n_Y, &n_loss);
            accumulated_epoch_loss += n_loss->value->data[0];

            // Backpropagation of gradients
            backward(n_loss);
            
            // Update weights
            update_params(arch->param_list, tensor_lr);

            // Zero gradient of the weights:
            zero_grad(arch->param_list);

            // Dispose computational graph and other stuff
            dispose_graph(n_loss);
            dispose_node(n_X);
            dispose_node(n_Y);
        }
        
        printf("Total Averaged Loss: %.4f\n", accumulated_epoch_loss / (num_batches * 1.0));
        printf("-------------------\n\n");

        // Store parameters in proper files
        store_tensor("./chckpts/bigram_ll1_W.txt", arch->ll1->n_W->value, 16);
        
        // Decay learning rate
        if (e % 10 == 0) {
            tensor_lr->data[0] /= 10;
        }
    }
    
    dispose_bigramarch(arch);
    dispose_bigram_dataset(dataset_size, dataset_X, dataset_Y);
}

void bigram_test() {
    // Define DNN architecture: 27 -> 27
    int token_size = 27;

    BigramArch *arch = NULL;
    create_bigramarch(token_size, token_size, &arch);

    // Initialize DNN from the stored parameters
    load_tensor("./chckpts/bigram_ll1_W.txt", arch->ll1->n_W->value);

    Tensor *X = NULL;
    create_tensor((int []){1, 27}, 2, &X);

    Node *n_X = NULL, *n_y_pred = NULL;

    for (int i = 0; i < 10; i++) {
        // Loop through the batch and one_hot encode X/Y
        printf(".");
        int idx = -1;
        while (idx != 26) {
            init_tensor(0.0, X);
            set_element(X, 1.0, 0, idx == -1 ? 26 : idx);

            // Convert tensor to node
            create_leaf(X, false, &n_X);
            
            // Forward pass through the DNN
            bigramforwad(arch, n_X, true, &n_y_pred);

            idx = pick_random_index(n_y_pred->value->data, n_y_pred->value->total_size, 2.5);

            if (idx != 26) {
                printf("%c", 'a' + idx);
            }
        }
        printf(".\n");
    }

    // Dispose computational graph and other stuff
    dispose_graph(n_y_pred);
    dispose_node(n_X);
    dispose_bigramarch(arch);
}

/////////////////////////////////////////////////////////////////////

int main(
    int argc,
    char *argv[]
) {
    setup_application(42);

    bigram_train("./names.txt");
    bigram_test();

    return 0;
}