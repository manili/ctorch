#include "ctorch.c"

typedef struct MLPArch {
    Queue *param_list;

    LinearLayer *embed;
    LinearLayer *ll1;
    ActivationLayer *al;
    LinearLayer *ll2;
    ActivationLayer *sm;
    
    CrossEntropyLoss *loss;
} MLPArch;

void create_mlparch(int in_feature_size, int out_feature_size, int block_size, MLPArch **out_arch) {
    *out_arch = (MLPArch *)malloc(sizeof(MLPArch));

    // Initialize parameters' list
    (*out_arch)->param_list = create_queue();

    int embed_size = 2;
    int hidden_size1 = embed_size * block_size;
    int hidden_size2 = 100;

    // Initialize DNN layers
    (*out_arch)->embed = linearlayer((*out_arch)->param_list, in_feature_size, embed_size, false);  // Input -> Hidden Layer 1
    (*out_arch)->ll1 = linearlayer((*out_arch)->param_list, hidden_size1, hidden_size2, true);      // Hidden Layer 1 -> Tanh
    (*out_arch)->al = activation_layer(tensor_tanh, grad_tensor_tanh);                              // Tanh -> Hidden Layer 2
    (*out_arch)->ll2 = linearlayer((*out_arch)->param_list, hidden_size2, out_feature_size, true);  // Hidden Layer 2 -> Softmax
    (*out_arch)->sm  = activation_layer(tensor_softmax, grad_tensor_softmax);                       // Softmax

    (*out_arch)->loss = crossentropyloss();                                                         // Cross Entropy Loss
}

void dispose_mlparch(MLPArch *arch) {
    dispose_linearlayer(arch->embed);
    dispose_linearlayer(arch->ll1);
    dispose_activationlayer(arch->al);
    dispose_linearlayer(arch->ll2);
    dispose_activationlayer(arch->sm);

    dispose_crossentropyloss(arch->loss);

    dispose_queue(arch->param_list);

    free(arch);
}

void mlpforwad(MLPArch *arch, Node *n_X, bool inference_mode, Node **n_y_pred) {
    Node *n_y1 = NULL, *n_y1_view = NULL, *n_y2 = NULL, *n_y3 = NULL, *n_y4 = NULL;
    forward_linearlayer(arch->embed, n_X, &n_y1);

    Tensor *dummy = NULL;
    int batch_size = n_y1->value->shape[0];
    int block_size = n_y1->value->shape[1];
    int embed_size = n_y1->value->shape[2];
    create_tensor_without_data((int []){batch_size, block_size * embed_size}, 2, &dummy);

    Node *n_dummy = NULL;
    create_leaf(dummy, false, &n_dummy);

    create_n_exec_op(tensor_view, grad_tensor_view, n_y1, n_dummy, &n_y1_view);
    forward_linearlayer(arch->ll1, n_y1_view, &n_y2);
    forward_activationlayer(arch->al, n_y2, &n_y3);
    forward_linearlayer(arch->ll2, n_y3, &n_y4);

    if (inference_mode) {
        forward_activationlayer_with_dim(arch->sm, n_y4, 1, n_y_pred);
    } else {
        *n_y_pred = n_y4;
    }
}

void mlploss(MLPArch *arch, Node *n_y_pred, Node *n_target, Node **n_loss) {
    forward_crossentropyloss(arch->loss, n_y_pred, n_target, 1, n_loss);
}

int countlines(
    const char *names_file
) {
    FILE *file = fopen(names_file, "r");
    if (file == NULL) {
        printf("Error: Unable to open file %s\n", names_file);
        exit(1);
    }

    int line_count = 0;
    char line[256];
    while (fgets(line, sizeof(line), file) != NULL) {
        int line_len = 0;
        while (line[line_len] >= 'a' && line[line_len] <= 'z') {
            line_count++;
            break;
        }
    }

    fclose(file);
 
    return line_count;
}

// Function to load the MLP dataset from names file
int load_mlp_dataset(
    const char *names_file,
    int block_size,
    char ***out_dataset_X,
    char **out_dataset_Y
) {
    FILE *file = fopen(names_file, "r");
    if (file == NULL) {
        printf("Error: Unable to open file %s\n", names_file);
        exit(1);
    }

    int total_dataset_size = 0;

    int line_idx = 0;
    int word_count = countlines(names_file);
    char line[256];
    int  lines_len[word_count];
    char lines[word_count][256];
    while (fgets(line, sizeof(line), file) != NULL) {
        int line_len = 1;
        lines[line_idx][0] = '.';
        while (line[line_len - 1] >= 'a' && line[line_len - 1] <= 'z') {
            lines[line_idx][line_len] = line[line_len - 1];
            line_len++;
        }
        lines[line_idx][line_len++] = '.';
        lines[line_idx][line_len] = '\0';
        lines_len[line_idx] = line_len;
        line_idx++;
        total_dataset_size += line_len - 1;
    }

    *out_dataset_X = (char **)malloc(total_dataset_size * sizeof(char *));
    *out_dataset_Y = (char *)malloc(total_dataset_size * sizeof(char));

    int dataset_idx = 0;
    for (int i = 0; i < word_count; i++) {
        char *word = lines[i];
        for (int j = 0; j < lines_len[i] - 1; j++) {
            (*out_dataset_X)[dataset_idx] = (char *)malloc(block_size * sizeof(char));
            int char_idx = 0;
            for (int k = j; k < j + block_size; k++) {
                int actual_idx = k - (block_size - 1);
                (*out_dataset_X)[dataset_idx][char_idx] = (actual_idx < 0) ? '.' : word[actual_idx];
                char_idx++;
            }
            (*out_dataset_Y)[dataset_idx] = word[j + 1];
            dataset_idx++;
        }
    }

    return total_dataset_size;
}

void dispose_mlp_dataset(
    int dataset_size,
    char **dataset_X,
    char *dataset_Y
) {
    for (int i = 0; i < dataset_size; i++) {
        free(dataset_X[i]);
        dataset_X[i] = NULL;
    }
    free(dataset_X);
    free(dataset_Y);
}

// Function to load a batch of MLP data into tensors X and Y
void load_mlp_batch(
    char **dataset_X,
    char *dataset_Y,
    int dataset_size,
    int block_size,
    int batch_size,
    int batch_idx,
    Tensor **out_X,
    Tensor **out_Y
) {
    int start_idx = batch_idx * batch_size;
    int actual_batch_size = (start_idx + batch_size < dataset_size) ? batch_size : dataset_size - start_idx;

    create_tensor((int []){actual_batch_size, block_size, 27}, 3, out_X);   // From a-z plus "."
    create_tensor((int []){actual_batch_size, 27}, 2, out_Y);               // From a-z plus "."
    init_tensor(0.0, *out_X);
    init_tensor(0.0, *out_Y);

    // Loop through the batch and one_hot encode X/Y
    for (int i = 0; i < actual_batch_size; i++) {
        int src_idx = start_idx + i;
        for (int j = 0; j < block_size; j++) {
            char chr_x = dataset_X[src_idx][j];
            int enc_x = (chr_x == '.') ? 26 : chr_x - 'a';
            set_element(*out_X, 1.0, i, j, enc_x);
        }

        char chr_y = dataset_Y[src_idx];
        int enc_y = (chr_y == '.') ? 26 : chr_y - 'a';
        set_element(*out_Y, 1.0, i, enc_y);
    }
}

void mlp_train(
    const char *mlp_train_names_file
) {
    // Define hyperparameters
    int block_size = 3;
    int tokens_count = 27;
    
    // Load the dataset
    char **dataset_X = NULL;
    char *dataset_Y = NULL;

    int dataset_size = load_mlp_dataset(mlp_train_names_file, block_size, &dataset_X, &dataset_Y);  // Adjust dataset size
    
    // Define hyperparameters
    int training_size = dataset_size;
    int batch_size = 64;
    int num_batches = ceil(training_size * 1.0 / batch_size);   // Number of batches in the epoch (adjust accordingly)
    int num_batches_to_print = 100;                             // Number of batches in the epoch to print result
    int epoch = 10;
    double lr = 0.1;    // Learning rate

    Tensor *tensor_lr = NULL;
    create_tensor_from_scalar(lr, &tensor_lr);  // Learning rate as a scalar tensor

    // Define DNN architecture
    MLPArch *arch = NULL;
    create_mlparch(tokens_count, tokens_count, block_size, &arch);

    // load_bigram_batch loads the Bigram batch of tokens
    for (int e = 1; e <= epoch; e++) {
        printf("Epoch: %d/%d\n\n", e, epoch);
        
        double accumulated_epoch_loss = 0.0;
        for (int b = 0; b < num_batches; b++) {
            // Load a batch of data (X, Y)
            Tensor *X = NULL, *Y = NULL;
            load_mlp_batch(dataset_X, dataset_Y, training_size, block_size, batch_size, b, &X, &Y);

            // Convert tensors to nodes
            Node *n_X = NULL, *n_Y = NULL;
            create_leaf(X, false, &n_X);
            create_leaf(Y, false, &n_Y);

            // Forward pass through the DNN
            Node* n_y_pred = NULL;
            mlpforwad(arch, n_X, false, &n_y_pred);

            // Loss calculation
            Node *n_loss = NULL;
            mlploss(arch, n_y_pred, n_Y, &n_loss);
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
        store_tensor("./chckpts/mlp_emb_W.txt", arch->embed->n_W->value, 16);
        store_tensor("./chckpts/mlp_ll1_W.txt", arch->ll1->n_W->value, 16);
        store_tensor("./chckpts/mlp_ll1_b.txt", arch->ll1->n_b->value, 16);
        store_tensor("./chckpts/mlp_ll2_W.txt", arch->ll2->n_W->value, 16);
        store_tensor("./chckpts/mlp_ll2_b.txt", arch->ll2->n_b->value, 16);

        // Decay learning rate
        if (e % 10 == 0) {
            tensor_lr->data[0] /= 10;    
        }
    }
    
    dispose_mlparch(arch);
    dispose_mlp_dataset(dataset_size, dataset_X, dataset_Y);
}

void mlp_test() {
    // Define DNN architecture: 27 -> 27
    int block_size = 3;
    int token_size = 27;

    MLPArch *arch = NULL;
    create_mlparch(token_size, token_size, block_size, &arch);

    // Initialize DNN from the stored parameters
    load_tensor("./chckpts/mlp_emb_W.txt", arch->embed->n_W->value);
    load_tensor("./chckpts/mlp_ll1_W.txt", arch->ll1->n_W->value);
    load_tensor("./chckpts/mlp_ll1_b.txt", arch->ll1->n_b->value);
    load_tensor("./chckpts/mlp_ll2_W.txt", arch->ll2->n_W->value);
    load_tensor("./chckpts/mlp_ll2_b.txt", arch->ll2->n_b->value);

    for (int i = 0; i < 10; i++) {
        // Loop through the batch and one_hot encode X/Y
        printf(".");
        int idx[3] = {-1};
        while (idx[2] != 26) {
            Tensor *X = NULL;
            create_tensor((int []){1, 3, 27}, 3, &X);
            init_tensor(0.0, X);
            set_element(X, 1.0, 0, 0, idx[0] == -1 ? 26 : idx[0]);
            set_element(X, 1.0, 0, 1, idx[1] == -1 ? 26 : idx[1]);
            set_element(X, 1.0, 0, 2, idx[2] == -1 ? 26 : idx[2]);

            // Convert tensor to node
            Node *n_X = NULL;
            create_leaf(X, false, &n_X);
            
            // Forward pass through the DNN
            Node *n_y_pred = NULL;
            mlpforwad(arch, n_X, true, &n_y_pred);

            idx[0] = idx[1];
            idx[1] = idx[2];
            idx[2] = pick_random_index(n_y_pred->value->data, n_y_pred->value->total_size, 2.7);

            if (idx[2] != 26) {
                printf("%c", 'a' + idx[2]);
            }
            
            // Dispose computational graph and other stuff
            dispose_graph(n_y_pred);
            dispose_node(n_X);
        }
        printf(".\n");
    }

    dispose_mlparch(arch);
}

/////////////////////////////////////////////////////////////////////

int main(
    int argc,
    char *argv[]
) {
    setup_application(42);
    
    mlp_train("./names.txt");
    mlp_test();

    return 0;
}