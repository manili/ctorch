#include "ctorch.c"

#define SIMPLE_TEST_DATASET_SIZE 64000
#define SIMPLE_TEST_FEATURE_SIZE 1

void custom_activation_pow2(
    OpArgs args,
    Tensor **out_grad_a,
    Tensor **out_grad_b,
    Tensor **out_tensor
) {
    Tensor *two = NULL;
    create_tensor_from_scalar(2.0, &two);
    tensor_pow(args.a, two, out_tensor);
    dispose_tensor(two, true);

    out_grad_a = NULL;
    out_grad_b = NULL;
}
void grad_custom_activation_pow2(
    OpArgs args,
    Tensor **out_grad_a,
    Tensor **out_grad_b,
    Tensor **out_tensor
) {
    Tensor *tmp_out_grad_a = NULL;
    Tensor *two = NULL;
    create_tensor_from_scalar(2.0, &two);
    tensor_mul(args.a, two, &tmp_out_grad_a);
    tensor_mul(tmp_out_grad_a, args.grad, out_grad_a);
    dispose_tensor(two, true);
    dispose_tensor(tmp_out_grad_a, true);
    
    out_grad_b = NULL;
    out_tensor = NULL;
}

typedef struct SimpleArch {
    Queue *param_list;

    LinearLayer *ll1;
    LinearLayer *ll2;
    ActivationLayer *al;
    
    MSELoss *loss;
} SimpleArch;

void simplearch(int input_size, int output_size, int batch_size, SimpleArch **out_arch) {
    *out_arch = (SimpleArch *)malloc(sizeof(SimpleArch));
    

    // Initialize parameters' list
    (*out_arch)->param_list = create_queue();

    // Initialize DNN layers
    (*out_arch)->ll1 = linearlayer((*out_arch)->param_list, input_size, 1, true);
    (*out_arch)->ll2 = linearlayer((*out_arch)->param_list, 1, output_size, true);
    (*out_arch)->al  = activation_layer(custom_activation_pow2, grad_custom_activation_pow2);

    (*out_arch)->loss = mseloss();
}

void simpleforwad(SimpleArch *arch, Node *n_X, Node **n_y_pred) {
    Node *n_y1 = NULL, *n_y2 = NULL;

    forward_linearlayer(arch->ll1, n_X, &n_y1);
    forward_linearlayer(arch->ll2, n_y1, &n_y2);
    forward_activationlayer(arch->al, n_y2, n_y_pred);
}

void simpleloss(SimpleArch *arch, Node *n_y_pred, Node *n_target, Node **n_loss) {
    forward_mseloss(arch->loss, n_y_pred, n_target, n_loss);
}

void load_simple_test_dataset(
    int dataset_size,
    int feature_size,
    Tensor **out_X,
    Tensor **out_Y
) {
    int X_shape[] = {dataset_size, feature_size};
    create_tensor(X_shape, 2, out_X);

    for (int i = 0; i < dataset_size; i++) {
        for (int j = 0; j < feature_size; j++) {
            double x = -32 + i * 0.001;
            x = x / 32.0;
            set_element(*out_X, x, i, j);
        }
    }
    
    int Y_shape[] = {dataset_size, feature_size};
    Tensor *Y = NULL;
    create_tensor(Y_shape, 2, out_Y);
    
    for (int i = 0; i < dataset_size; i++) {
        for (int j = 0; j < feature_size; j++) {
            double y = -32 + i * 0.001;
            y = y / 32.0;
            y = y * y;
            set_element(*out_Y, y, i, j);
        }
    }
}

// Function to load a batch of simple_test data into tensors X and Y
void load_simple_test_batch(
    Tensor *X_dataset,
    Tensor *Y_dataset,
    int batch_idx,
    int batch_size,
    int feature_size,
    bool shuffle,
    Tensor *io_X,
    Tensor *io_Y
) {
    int start_idx = batch_idx * batch_size;

    // Loop through the batch
    for (int i = 0; i < batch_size; i++) {
        int data_idx = start_idx + i;
        for (int j = 0; j < feature_size; j++) {
            int multi_dim_idx[2] = {i, j};
            int idx = get_flat_index(X_dataset, multi_dim_idx);
            set_element(io_X, X_dataset->data[idx], i, j);
            set_element(io_Y, Y_dataset->data[idx], i, j);
        }
    }

    if (shuffle) {
        shuffle_data(io_X, io_Y);
    }
}

void simple_test() {
    int dataset_size = SIMPLE_TEST_DATASET_SIZE;
    int feature_size = SIMPLE_TEST_FEATURE_SIZE;

    int batch_size = 64000;
    int num_batches = 1;
    int epoch = 1000;
    double lr = 0.01;

    Tensor *X_dataset = NULL, *Y_dataset = NULL;
    load_simple_test_dataset(dataset_size, feature_size, &X_dataset, &Y_dataset);

    // Placeholder for batch input and labels
    int X_shape[] = {batch_size, feature_size};
    int Y_shape[] = {batch_size, feature_size};
    Tensor *X = NULL, *Y = NULL;
    create_tensor(X_shape, 2, &X);
    create_tensor(Y_shape, 2, &Y);

    Tensor *tensor_lr = NULL;
    create_tensor_from_scalar(lr, &tensor_lr);

    // Initialize DNN layers
    SimpleArch *arch = NULL;
    simplearch(feature_size, feature_size, batch_size, &arch);

    for (int i = 1; i < epoch + 1; i++) {
        printf("Epoch: %d/%d\n\n", i, epoch);

        if (i % 60 == 0 && tensor_lr->data[0] > 0.0001) {
            tensor_lr->data[0] /= 10;
        }

        for (int j = 0; j < num_batches; j++) {
            load_simple_test_batch(X_dataset, Y_dataset, j, batch_size, feature_size, true, X, Y);

            // Convert Tensors into Leaf Nodes
            Node *n_X = NULL, *n_Y= NULL;
            create_leaf(X, false, &n_X);
            create_leaf(Y, false, &n_Y);

            // Forward pass through the DNN
            Node *n_y_pred = NULL;
            simpleforwad(arch, n_X, &n_y_pred);

            // Loss calculation
            Node *n_loss = NULL;
            simpleloss(arch, n_y_pred, n_Y, &n_loss);

            // Backpropagation of gradients
            backward(n_loss);
            
            // Update weights
            update_params(arch->param_list, tensor_lr);

            // Zero gradient of the weights:
            zero_grad(arch->param_list);

            // Every 100 batches, print the loss (MSE)
            if (j % 100 == 0) {
                printf("Batch %d/%d:\n", j, num_batches);
                printf("W1:\n");
                print_info(arch->ll1->n_W->value);
                printf("W2:\n");
                print_info(arch->ll2->n_W->value);
                printf("Loss:\n");
                print_info_with_precision(n_loss->value, 16);  // Print the loss tensor
                printf("-------------------\n");
            }

            // Dispose computational graph and other stuff
            dispose_graph(n_loss);
            dispose_node(n_X);
            dispose_node(n_Y);
        }
    }
}

/////////////////////////////////////////////////////////////////////

int main(
    int argc,
    char *argv[]
) {
    setup_application(42);

    simple_test();

    return 0;
}