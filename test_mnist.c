#include "ctorch.c"

typedef struct MNISTArch {
    Queue *param_list;

    LinearLayer *ll1;
    ActivationLayer *al1;
    LinearLayer *ll2;
    ActivationLayer *al2;
    LinearLayer *ll3;
    ActivationLayer *al3;
    LinearLayer *ll4;
    ActivationLayer *sm;
    
    CrossEntropyLoss *loss;
} MNISTArch;

void create_mnistarch(int in_feature_size, int out_feature_size, MNISTArch **out_arch) {
    *out_arch = (MNISTArch *)malloc(sizeof(MNISTArch));

    // Initialize parameters' list
    (*out_arch)->param_list = create_queue();

    int hidden_size1 = 512;
    int hidden_size2 = 256;
    int hidden_size3 = 128;

    // Initialize DNN layers
    (*out_arch)->ll1 = linearlayer((*out_arch)->param_list, in_feature_size, hidden_size1, true);   // Input -> Hidden Layer 1
    (*out_arch)->al1 = activation_layer(tensor_relu, grad_tensor_relu);                             // Activation after Layer 1
    (*out_arch)->ll2 = linearlayer((*out_arch)->param_list, hidden_size1, hidden_size2, true);      // Hidden Layer 1 -> Hidden Layer 2
    (*out_arch)->al2 = activation_layer(tensor_relu, grad_tensor_relu);                             // Activation after Layer 2
    (*out_arch)->ll3 = linearlayer((*out_arch)->param_list, hidden_size2, hidden_size3, true);      // Hidden Layer 2 -> Hidden Layer 3
    (*out_arch)->al3 = activation_layer(tensor_relu, grad_tensor_relu);                             // Activation after Layer 3
    (*out_arch)->ll4 = linearlayer((*out_arch)->param_list, hidden_size3, out_feature_size, true);  // Hidden Layer 3 -> Softmax
    (*out_arch)->sm  = activation_layer(tensor_softmax, grad_tensor_softmax);                       // Softmax

    (*out_arch)->loss = crossentropyloss();                                                         // Cross Entropy Loss
}

void dispose_mnistarch(MNISTArch *arch) {
    dispose_linearlayer(arch->ll1);
    dispose_activationlayer(arch->al1);
    dispose_linearlayer(arch->ll2);
    dispose_activationlayer(arch->al2);
    dispose_linearlayer(arch->ll3);
    dispose_activationlayer(arch->al3);
    dispose_linearlayer(arch->ll4);
    dispose_activationlayer(arch->sm);

    dispose_crossentropyloss(arch->loss);

    dispose_queue(arch->param_list);

    free(arch);
}

void mnistforwad(MNISTArch *arch, Node *n_X, bool inference_mode, Node **n_y_pred) {
    Node *n_y1 = NULL, *n_y2 = NULL, *n_y3 = NULL, *n_y4 = NULL;
    Node *n_y5 = NULL, *n_y6 = NULL, *n_y7 = NULL;

    forward_linearlayer(arch->ll1, n_X, &n_y1);
    forward_activationlayer(arch->al1, n_y1, &n_y2);

    forward_linearlayer(arch->ll2, n_y2, &n_y3);
    forward_activationlayer(arch->al2, n_y3, &n_y4);
    
    forward_linearlayer(arch->ll3, n_y4, &n_y5);
    forward_activationlayer(arch->al3, n_y5, &n_y6);

    forward_linearlayer(arch->ll4, n_y6, &n_y7);
    
    if (inference_mode) {
        forward_activationlayer_with_dim(arch->sm, n_y7, 1, n_y_pred);
    } else {
        *n_y_pred = n_y7;
    }
}

void mnistloss(MNISTArch *arch, Node *n_y_pred, Node *n_target, Node **n_loss) {
    forward_crossentropyloss(arch->loss, n_y_pred, n_target, 1, n_loss);
}

// Function to load the MNIST dataset from a CSV file
void load_mnist_dataset(
    const char *mnist_csv_file,
    int num_images,
    int image_size,
    double ***out_mnist_images, // Array of size [NUM_IMAGES][784] for training images
    int **out_mnist_labels      // Array of size [NUM_IMAGES] for training labels
) {
    FILE *file = fopen(mnist_csv_file, "r");
    if (file == NULL) {
        printf("Error: Unable to open file %s\n", mnist_csv_file);
        exit(1);
    }

    // Allocate memory for the mnist_images array (2D array)
    *out_mnist_images = (double **)malloc(num_images * sizeof(double *));
    for (int i = 0; i < num_images; i++) {
        (*out_mnist_images)[i] = (double *)malloc(image_size * sizeof(double));
    }

    // Allocate memory for the mnist_labels array (1D array)
    *out_mnist_labels = (int *)malloc(num_images * sizeof(int));

    // Read each line in the CSV file (assuming the first column is the label)
    char line[4096];  // Large enough to hold a line with 785 values (label + 784 pixels)
    int image_idx = 0;

    while (fgets(line, sizeof(line), file)) {
        char *token = strtok(line, ",");  // Tokenize the line by commas

        // First token is the label
        (*out_mnist_labels)[image_idx] = atoi(token);

        // Read the next 784 tokens for pixel values
        for (int i = 0; i < image_size; i++) {
            token = strtok(NULL, ",");
            if (token != NULL) {
                (*out_mnist_images)[image_idx][i] =  atof(token) / 255.0; // Transform pixel [0, 255] -> [0, 1.0]
            }
        }

        image_idx++;

        // Stop if we have loaded enough images (for small datasets)
        if (image_idx >= num_images) {
            break;
        }
    }

    fclose(file);
}

void dispose_mnist_dataset(
    int num_images,
    int image_size,
    double **mnist_images,
    int *mnist_labels
) {
    for (int i = 0; i < num_images; i++) {
        free(mnist_images[i]);
        mnist_images[i] = NULL;
    }
    free(mnist_images);
    mnist_images = NULL;

    free(mnist_labels);
    mnist_labels = NULL;
}

// Function to load a batch of MNIST data into tensors X and Y
void load_mnist_batch(
    double **mnist_images,
    int *mnist_labels,
    int dataset_size,
    int batch_idx,
    int batch_size,
    Tensor **out_X,
    Tensor **out_Y
) {
    int start_idx = batch_idx * batch_size;
    int actual_batch_size = (start_idx + batch_size < dataset_size) ? batch_size : dataset_size - (start_idx + 1);

    create_tensor((int[2]){actual_batch_size, 784}, 2, out_X);  // Create (batch_size, 784) tensor
    create_tensor((int[2]){actual_batch_size,  10}, 2, out_Y);  // Create (batch_size, 10) tensor

    // Loop through the batch
    for (int i = 0; i < actual_batch_size; i++) {
        int src_idx = start_idx + i + 1;

        // Load the image and flatten it into a 784-length vector
        for (int j = 0; j < 784; j++) {
            set_element(*out_X, mnist_images[src_idx][j], i, j);
        }

        // Load the label and one-hot encode it into a 10-length vector
        double one_hot_label[10] = {0};
        for (int i = 0; i < 10; i++) {
            one_hot_label[i] = (i == mnist_labels[src_idx]) ? 1.0 : 0.0;
        }

        for (int j = 0; j < 10; j++) {
            set_element(*out_Y, one_hot_label[j], i, j);
        }
    }
}

void mnist_train(
    const char *mnist_train_csv_file
) {
    double **mnist_images = NULL;
    int *mnist_labels = NULL;

    int dataset_size = 20000;   // Adjust dataset size
    int image_size = 784;       // Flattened MNIST image (28 * 28)
    int label_size = 10;        // MNIST has 10 classes (digits 0-9)

    // Load the MNIST dataset from the CSV file
    load_mnist_dataset(mnist_train_csv_file, dataset_size, image_size, &mnist_images, &mnist_labels);

    // Define hyperparameters
    int training_size = dataset_size;
    int batch_size = 64;
    int num_batches = ceil(training_size * 1.0 / batch_size);   // Number of batches in the epoch (adjust accordingly)
    int num_batches_to_print = 100;                             // Number of batches in the epoch to print result
    int epoch = 10;
    double lr = 0.1;    // Learning rate

    // Define DNN architecture: 784 -> 512 -> 128 -> 10
    int input_size = image_size;
    int output_size = label_size;

    Tensor *tensor_lr = NULL;
    create_tensor_from_scalar(lr, &tensor_lr);  // Learning rate as a scalar tensor

    // Create DNN layers
    MNISTArch *arch = NULL;
    create_mnistarch(input_size, output_size, &arch);

    // load_mnist_batch loads the MNIST batch of images and labels
    for (int e = 1; e <= epoch; e++) {
        printf("Epoch: %d/%d\n\n", e, epoch);
        
        double accumulated_epoch_loss = 0.0;
        for (int b = 0; b < num_batches; b++) {
            // Load a batch of data (X, Y)
            Tensor *X = NULL, *Y = NULL;
            load_mnist_batch(mnist_images, mnist_labels, training_size, b, batch_size, &X, &Y);

            // Convert tensors to node
            Node *n_X = NULL, *n_Y = NULL;
            create_leaf(X, false, &n_X);
            create_leaf(Y, false, &n_Y);

            // Forward pass through the DNN
            Node *n_y_pred = NULL;
            mnistforwad(arch, n_X, false, &n_y_pred);

            // Loss calculation
            Node *n_loss = NULL;
            mnistloss(arch, n_y_pred, n_Y, &n_loss);
            accumulated_epoch_loss += n_loss->value->data[0];
            
            // Backpropagation of gradients
            backward(n_loss);

            // Update weights
            update_params(arch->param_list, tensor_lr);            

            // Zero gradient of the weights:
            zero_grad(arch->param_list);

            // Print the loss
            if ((b + 1) % num_batches_to_print == 0) {
                printf("Batch %d/%d - Loss: %.4f\n\n", b + 1, num_batches, accumulated_epoch_loss / (b + 1.0));
            }

            // Dispose computational graph and other stuff
            dispose_graph(n_loss);
            dispose_node(n_X);
            dispose_node(n_Y);
        }
        
        printf("Total Averaged Loss: %.4f\n", accumulated_epoch_loss / (num_batches * 1.0));
        printf("-------------------\n\n");

            
        // Store parameters in proper files at the end of each epoch
        store_tensor("./chckpts/mnist_ll1_W.txt", arch->ll1->n_W->value, 16);
        store_tensor("./chckpts/mnist_ll1_b.txt", arch->ll1->n_b->value, 16);
        store_tensor("./chckpts/mnist_ll2_W.txt", arch->ll2->n_W->value, 16);
        store_tensor("./chckpts/mnist_ll2_b.txt", arch->ll2->n_b->value, 16);
        store_tensor("./chckpts/mnist_ll3_W.txt", arch->ll3->n_W->value, 16);
        store_tensor("./chckpts/mnist_ll3_W.txt", arch->ll3->n_b->value, 16);
        store_tensor("./chckpts/mnist_ll4_W.txt", arch->ll4->n_W->value, 16);
        store_tensor("./chckpts/mnist_ll4_b.txt", arch->ll4->n_b->value, 16);
        
        // Decay learning rate
        if (e % 3 == 0) {
            tensor_lr->data[0] /= 10;
        }
    }
    
    dispose_mnistarch(arch);
    dispose_mnist_dataset(dataset_size, image_size, mnist_images, mnist_labels);
}

void mnist_eval(
    const char *mnist_eval_csv_file
) {
    double **mnist_images = NULL;
    int *mnist_labels = NULL;

    int dataset_size = 10000;   // Adjust dataset size
    int image_size = 784;       // Flattened MNIST image (28 * 28)
    int label_size = 10;        // MNIST has 10 classes (digits 0-9)

    // Load the MNIST dataset from the CSV file
    load_mnist_dataset(mnist_eval_csv_file, dataset_size, image_size, &mnist_images, &mnist_labels);

    // Define hyperparameters
    int eval_size = dataset_size;
    int batch_size = 64;
    int num_batches = ceil(eval_size * 1.0 / batch_size);   // Number of batches in the epoch (adjust accordingly)

    // Define DNN architecture: 784 -> 512 -> 128 -> 10
    int input_size = image_size;
    int output_size = label_size;

    // Create DNN layers
    MNISTArch *arch = NULL;
    create_mnistarch(input_size, output_size, &arch);

    // Initialize DNN from the stored parameters
    load_tensor("./chckpts/mnist_ll1_W.txt", arch->ll1->n_W->value);
    load_tensor("./chckpts/mnist_ll1_b.txt", arch->ll1->n_b->value);
    load_tensor("./chckpts/mnist_ll2_W.txt", arch->ll2->n_W->value);
    load_tensor("./chckpts/mnist_ll2_b.txt", arch->ll2->n_b->value);
    load_tensor("./chckpts/mnist_ll3_W.txt", arch->ll3->n_W->value);
    load_tensor("./chckpts/mnist_ll3_b.txt", arch->ll3->n_b->value);
    load_tensor("./chckpts/mnist_ll4_W.txt", arch->ll4->n_W->value);
    load_tensor("./chckpts/mnist_ll4_b.txt", arch->ll4->n_b->value);

    double accumulated_epoch_loss = 0.0;
    for (int b = 0; b < num_batches; b++) {
        // Load a batch of data (X, Y)
        Tensor *X = NULL, *Y = NULL;
        load_mnist_batch(mnist_images, mnist_labels, eval_size, b, batch_size, &X, &Y);

        // Convert tensors to node
        Node *n_X = NULL, *n_Y = NULL;
        create_leaf(X, false, &n_X);
        create_leaf(Y, false, &n_Y);

        // Forward pass through the DNN
        Node *n_y_pred = NULL;
        mnistforwad(arch, n_X, false, &n_y_pred);

        // Loss calculation
        Node *n_loss = NULL;
        mnistloss(arch, n_y_pred, n_Y, &n_loss);
        accumulated_epoch_loss += n_loss->value->data[0];

        // Dispose computational graph and other stuff
        dispose_graph(n_loss);
        dispose_node(n_X);
        dispose_node(n_Y);
    }
    
    printf("Total Averaged Loss: %.4f\n", accumulated_epoch_loss / (num_batches * 1.0));
    
    dispose_mnistarch(arch); 
    dispose_mnist_dataset(dataset_size, image_size, mnist_images, mnist_labels);
}

/////////////////////////////////////////////////////////////////////

int main(
    int argc,
    char *argv[]
) {
    setup_application(42);

    mnist_train("./mnist_train_small.csv");
    mnist_eval("./mnist_test.csv");

    return 0;
}