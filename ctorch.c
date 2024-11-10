#include <math.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>
#include <stdbool.h>

/////////////////////////////////////////////////////////////////////

// Structure to hold an n-dimensional tensor with index mapping
typedef struct {
    int *shape;        // Array to hold the size of each dimension
    int dimensions;    // Number of dimensions
    double *data;      // Shared data array
    int total_size;    // Total number of elements in the tensor
    int *strides;      // Array to hold the stride of each dimension
    int *org_strides;  // Array to hold the stride of each dimension regardless of whether it is a shared transposed tensor or not
} Tensor;

// Function to calculate the total size of the tensor
int calculate_total_size(
    int *shape,
    int dimensions
) {
    int total_size = 1;
    for (int i = 0; i < dimensions; i++) {
        total_size *= shape[i];
    }
    return total_size;
}

// Function to calculate strides for a tensor based on its shape
void calculate_strides(
    int *shape,
    int dimensions,
    int *out_strides
) {
    out_strides[dimensions - 1] = 1;
    for (int i = dimensions - 2; i >= 0; i--) {
        out_strides[i] = out_strides[i + 1] * shape[i + 1];
    }
}

// Function to free the tensor's memory
void dispose_tensor(
    Tensor *tensor,
    bool free_data
) {
    if (tensor == NULL) return;

    free(tensor->shape); tensor->shape = NULL;
    free(tensor->strides); tensor->strides = NULL;
    free(tensor->org_strides); tensor->org_strides = NULL;
    if (free_data) { free(tensor->data); tensor->data = NULL; }
    free(tensor);
}

// Function to initialize a tensor with a given shape
void create_tensor(
    int *shape,
    int dimensions,
    Tensor **out_tensor
) {
    int total_size = calculate_total_size(shape, dimensions);

    // Allocate memory and initialization
    *out_tensor = (Tensor *)malloc(sizeof(Tensor));
    (*out_tensor)->shape = (int *)malloc(dimensions * sizeof(int));
    (*out_tensor)->strides = (int *)malloc(dimensions * sizeof(int));
    (*out_tensor)->org_strides = (int *)malloc(dimensions * sizeof(int));
    (*out_tensor)->data = (double *)malloc(total_size * sizeof(double));
    
    (*out_tensor)->dimensions = dimensions;
    (*out_tensor)->total_size = total_size;
    
    memcpy((*out_tensor)->shape, shape, dimensions * sizeof(int));
    calculate_strides(shape, dimensions, (*out_tensor)->strides);
    calculate_strides(shape, dimensions, (*out_tensor)->org_strides);
}

// Function to initialize a tensor with a given shape
void deep_copy_tensor(
    Tensor *tensor,
    Tensor **out_tensor
) {
    double *data = tensor->data;
    int *shape = tensor->shape;
    int dimensions = tensor->dimensions;
    int total_size = calculate_total_size(shape, dimensions);

    // Allocate memory and initialization
    *out_tensor = (Tensor *)malloc(sizeof(Tensor));
    (*out_tensor)->shape = (int *)malloc(dimensions * sizeof(int));
    (*out_tensor)->strides = (int *)malloc(dimensions * sizeof(int));
    (*out_tensor)->org_strides = (int *)malloc(dimensions * sizeof(int));
    (*out_tensor)->data = (double *)malloc(total_size * sizeof(double));

    (*out_tensor)->dimensions = dimensions;
    (*out_tensor)->total_size = total_size;
    
    memcpy((*out_tensor)->data, data, total_size * sizeof(double));
    memcpy((*out_tensor)->shape, shape, dimensions * sizeof(int));
    calculate_strides(shape, dimensions, (*out_tensor)->strides);
    calculate_strides(shape, dimensions, (*out_tensor)->org_strides);
}

void create_tensor_from_scalar(
    double value,
    Tensor **out_tensor
) {
    int tensor_shape[] = {1, 1};
    create_tensor(tensor_shape, 2, out_tensor);
    (*out_tensor)->data[0] = value;
}

void create_tensor_from_tensor(
    Tensor *tensor,
    int index,
    Tensor **out_tensor
) {
    int new_tensor_shape[] = {1, 1};
    create_tensor(new_tensor_shape, 2, out_tensor);
    free((*out_tensor)->data);
    (*out_tensor)->data = &tensor->data[index];
}

// Initialize all elements to default_vlaue
void init_tensor(
    double default_vlaue,
    Tensor *tensor
) {
    for (int i = 0; i < tensor->total_size; i++) {
        tensor->data[i] = default_vlaue;
    }
}

// Initialize all elements randomly between -sqrt(k) and sqrt(k)
void init_tensor_rand(
    double k,       // Parameter for controlling the range
    Tensor *tensor
) {
    double range = sqrt(k);  // Calculate sqrt(k) once for efficiency

    // Fill each element of the tensor with a random double between -sqrt(k) and sqrt(k)
    for (int i = 0; i < tensor->total_size; i++) {
        // Generate a uniform random number in [-sqrt(k), sqrt(k)]
        double random_value = ((double)rand() / RAND_MAX) * 2 - 1; // Uniformly in [-1, 1]
        tensor->data[i] = random_value * range; // Scale to [-sqrt(k), sqrt(k)]
    }
}

// Function to store tensor values in a file with specified precision
void store_tensor(
    const char *filename,
    Tensor *tensor,
    int precision
) {
    FILE *file = fopen(filename, "w");
    if (file == NULL) {
        printf("Error: Unable to open file %s\n", filename);
        exit(1);
    }

    // Format string to control precision, e.g., "%.3f\n" for 3 decimal places
    char format[10];
    snprintf(format, sizeof(format), "%%.%df\n", precision);

    // Write each element of the tensor's data to the file with the specified precision
    for (int i = 0; i < tensor->total_size; i++) {
        fprintf(file, format, tensor->data[i]);
    }

    fclose(file);
}

// Initialize all elements from a previous specified values in a file
void load_tensor(
    const char *filename,
    Tensor *tensor
) {
    FILE *file = fopen(filename, "r");
    if (file == NULL) {
        printf("Error: Unable to open file %s\n", filename);
        exit(1);
    }

    // Read each line in the file
    char line[4096];  // Large enough to hold a line with 785 values (label + 784 pixels)
    int idx = 0;

    while (fgets(line, sizeof(line), file)) {
        char *token = strtok(line, "\n");  // Tokenize the line by commas

        // First token is the label
        tensor->data[idx++] = atof(token);
    }

    fclose(file);
}

// Function to calculate the flattened index for the tensor from multi-dimensional indices
int get_flat_index(
    Tensor *tensor,
    int *indices
) {
    int flat_index = 0;

    // Iterate through each dimension
    for (int i = 0; i < tensor->dimensions; i++) {
        int index = indices[i];
        if (index >= tensor->shape[i] || index < 0) {
            fprintf(stderr, "Error: Index out of bounds for dimension %d.\n", i);
            exit(EXIT_FAILURE);
        }

        // Use the precomputed stride to calculate the flat index
        flat_index += index * tensor->strides[i];
    }

    return flat_index;
}

// Function to calculate the multi-dimensional indices from a flattened index
void get_multi_dimensional_index(
    Tensor *tensor,
    int flat_index,
    int *out_multi_dim_indices
) {
    // Ensure the flat index is within bounds
    if (flat_index < 0 || flat_index >= tensor->total_size) {
        fprintf(stderr, "Error: Flattened index out of bounds.\n");
        exit(EXIT_FAILURE);
    }

    // Calculate the indices for each dimension using strides
    for (int i = 0; i < tensor->dimensions; i++) {
        // Determine the index for this dimension using the corresponding precomputed stride
        out_multi_dim_indices[i] = flat_index / tensor->org_strides[i];
        
        // Update the flat_index to the remainder for the next dimensions
        flat_index %= tensor->org_strides[i];
    }
}

// Function to get an element from the tensor using multi-dimensional indices
double get_element(
    Tensor *tensor,
    ...
) {
    va_list args;
    int *index_in_d = (int *)malloc(tensor->dimensions * sizeof(int));
    
    va_start(args, tensor);
    for (int i = 0; i < tensor->dimensions; i++) {
        int index = va_arg(args, int);
        index_in_d[i] = index;
    }
    va_end(args);

    int flat_index = get_flat_index(tensor, index_in_d);

    // Free the index_in_d array
    free(index_in_d);

    return tensor->data[flat_index];
}

// Function to set an element in the tensor using multi-dimensional indices
void set_element(
    Tensor *tensor,
    double value,
    ...
) {
    va_list args;
    int *index_in_d = (int *)malloc(tensor->dimensions * sizeof(int));

    va_start(args, value);
    for (int i = 0; i < tensor->dimensions; i++) {
        int index = va_arg(args, int);
        index_in_d[i] = index;
    }
    va_end(args);

    int flat_index = get_flat_index(tensor, index_in_d);

    tensor->data[flat_index] = value;

    // Free the index_in_d array
    free(index_in_d);
}

// Function to compare two tensors for equality
bool equal(
    Tensor *a,
    Tensor *b
) {
    // Check if the number of dimensions is the same
    if (a->dimensions != b->dimensions) {
        return false;
    }

    // Check if the shape of each dimension is the same
    for (int i = 0; i < a->dimensions; i++) {
        if (a->shape[i] != b->shape[i]) {
            return false;
        }
    }

    // Check if the data in each tensor is the same
    int *indices = (int *)malloc(a->dimensions * sizeof(int));
    for (int i = 0; i < a->total_size; i++) {
        // Get the multi-dimensional index for the current flat index
        // Multi-dim is the common index type among A and AT
        get_multi_dimensional_index(a, i, indices);
        int flat_index_a = get_flat_index(a, indices);
        int flat_index_b = get_flat_index(b, indices);

        // Compare the data values at the calculated flat indices
        if (a->data[flat_index_a] != b->data[flat_index_b]) {
            free(indices);
            return false;
        }
    }

    // Free allocated memory
    free(indices);

    // If all checks passed, the tensors are equal
    return true;
}

// Function to compare two tensors for equality except their data
bool equal_exclude_data(
    Tensor *a,
    Tensor *b
) {
    // Check if the number of dimensions is the same
    if (a->dimensions != b->dimensions) {
        return false;
    }

    // Check if the shape of each dimension is the same
    for (int i = 0; i < a->dimensions; i++) {
        if (a->shape[i] != b->shape[i]) {
            return false;
        }
    }

    // Check if the strides of each dimension is the same
    for (int i = 0; i < a->dimensions; i++) {
        if (a->strides[i] != b->strides[i]) {
            return false;
        }
    }

    // Check if the org_strides of each dimension is the same
    for (int i = 0; i < a->dimensions; i++) {
        if (a->org_strides[i] != b->org_strides[i]) {
            return false;
        }
    }

    // If all checks passed, the tensors are equal
    return true;
}

// Print double number with specified precision
void __print_double(
    double number,
    int precision
) {
    // Format string to control precision dynamically
    char format[50];
    snprintf(format, sizeof(format), "%%.%df", precision);

    // Print the number with the specified precision
    printf(format, number);
}

// Print info of a tensor
void __print_info_helper(
    Tensor *tensor,
    int precision,
    int dim,
    int* index
) {
    if (tensor->dimensions == 1 && tensor->shape[0] == 1) {
        printf("[");
        __print_double(tensor->data[0], precision);
        printf("]");
    } else if (dim < tensor->dimensions - 1) {
        printf("[");
        for (int i = 0; i < tensor->shape[dim]; i++) {
            index[dim] = i; 
            __print_info_helper(tensor, precision, dim + 1, index);
            if (i < tensor->shape[dim] - 1) {
                printf(",\n");
                for (int j = 0; j < tensor->dimensions - 2 - dim; j++) {
                    printf("\n");
                }
            }
        }
        printf("]");
    } else {
        printf("[");
        for (int i = 0; i < tensor->shape[dim]; i++) {
            index[dim] = i;
            int flat_idx = get_flat_index(tensor, index);

            if (i == tensor->shape[dim] - 1) {
                __print_double(tensor->data[flat_idx], precision);
            } else {
                __print_double(tensor->data[flat_idx], precision);
                printf(", ");
            }
        }
        printf("]");
    }
}

// Print info of a tensor
void __print_info(
    Tensor *tensor,
    int precision
) {
    int *index = (int *)malloc(tensor->dimensions * sizeof(int));
    for (int i = 0; i < tensor->dimensions; i++) {
        index[i] = 0;
    }
    
    printf("[");
    for (int i = 0; i < tensor->shape[0]; i++) {
        index[0] = i;
        __print_info_helper(tensor, precision, 1, index);
        if (i < tensor->shape[0] - 1) {
            printf(",\n");
            for (int j = 0; j < tensor->dimensions - 2; j++) {
                printf("\n");
            }
        }
    }
    printf("]\n\n");


    printf("(");
    for (int i = 0; i < tensor->dimensions; i++) {
        if (i < tensor->dimensions - 1) {
            printf("%d,", tensor->shape[i]);
        } else {
            printf("%d", tensor->shape[i]);
        }
    }
    printf(")\n\n");
    
    free(index);
}

void print_info(
    Tensor *tensor
) {
    __print_info(tensor, 4);
}

void print_info_with_precision(
    Tensor *tensor,
    int precision
) {
    __print_info(tensor, precision);
}

void print(
    int *arr,
    int count
) {
    for (int i = 0; i < count; i++) {
        printf("%d", arr[i]);
        if (i < count - 1) {
            printf(", ");
        }
    }
    printf("\n\n");
}

// Function to broadcast tensors so that they would align each other
void broadcast(
    Tensor *a,
    Tensor *b,
    int num_preserved_dims_a,
    int *preserved_dims_a,
    int num_preserved_dims_b,
    int *preserved_dims_b,
    Tensor **out_a,
    Tensor **out_b
) {
    // Determine the maximum number of dimensions
    int max_dims = (a->dimensions > b->dimensions) ? a->dimensions : b->dimensions;
    int offset_dims = (a->dimensions > b->dimensions) ? a->dimensions - b->dimensions : b->dimensions - a->dimensions;

    // Allocate memory for broadcasted shapes
    int *broadcast_shape_a = (int *)malloc(max_dims * sizeof(int));
    int *broadcast_shape_b = (int *)malloc(max_dims * sizeof(int));

    // Arrays to store preserved situation of dimensions (initialize all as false)
    bool *state_dims_a = (bool *)calloc(max_dims, sizeof(bool));
    bool *state_dims_b = (bool *)calloc(max_dims, sizeof(bool));

    // Identify preserved dimensions
    for (int i = 0; i < num_preserved_dims_a; i++) {
        int dim_to_preserve = preserved_dims_a[i];
        if (a->dimensions < max_dims) {
            state_dims_a[offset_dims + dim_to_preserve] = true;
        } else {
            state_dims_a[dim_to_preserve] = true;
        }
    }
    
    for (int i = 0; i < num_preserved_dims_b; i++) {
        int dim_to_preserve = preserved_dims_b[i];
        if (b->dimensions < max_dims) {
            state_dims_b[offset_dims + dim_to_preserve] = true;
        } else {
            state_dims_b[dim_to_preserve] = true;
        }
    }

    // Fill in the shapes starting from the leftmost dimension
    for (int i = 0; i < max_dims; i++) {
        int dim_a = (i >= max_dims - a->dimensions) ? a->shape[i - (max_dims - a->dimensions)] : 1;
        int dim_b = (i >= max_dims - b->dimensions) ? b->shape[i - (max_dims - b->dimensions)] : 1;

        // Determine the broadcasted dimension size, only if the dimension is not preserved
        if (state_dims_a[i]) {
            broadcast_shape_a[i] = dim_a;
        } else {
            // Apply regular broadcasting rules
            if (dim_a == dim_b) {
                broadcast_shape_a[i] = dim_a;
            } else if (dim_a > 1 && dim_b == 1) {
                broadcast_shape_a[i] = dim_a;
            } else if (dim_a == 1) {
                broadcast_shape_a[i] = dim_b;
            } else {
                fprintf(stderr, "Error: Tensors are not broadcastable.\n");
                exit(EXIT_FAILURE);
            }
        }

        if (state_dims_b[i]) {
            broadcast_shape_b[i] = dim_b;
        } else {
            // Apply regular broadcasting rules
            if (dim_a == dim_b) {
                broadcast_shape_b[i] = dim_b;
            } else if (dim_b > 1 && dim_a == 1) {
                broadcast_shape_b[i] = dim_b;
            } else if (dim_b == 1) {
                broadcast_shape_b[i] = dim_a;
            } else {
                fprintf(stderr, "Error: Tensors are not broadcastable.\n");
                exit(EXIT_FAILURE);
            }
        }
    }

    // Create the output tensors with the broadcasted shape
    create_tensor(broadcast_shape_a, max_dims, out_a);
    create_tensor(broadcast_shape_b, max_dims, out_b);

    // Broadcast tensor a and fill in tensor out_a
    int offset_a = max_dims - a->dimensions;
    int *src_idx_a = (int *)malloc(a->dimensions * sizeof(int));
    int *dest_idx_a = (int *)malloc((*out_a)->dimensions * sizeof(int));

    for (int i = 0; i < (*out_a)->total_size; i++) {
        get_multi_dimensional_index(*out_a, i, dest_idx_a);

        for (int j = offset_a; j < max_dims; j++) {
            int orig_idx = j - offset_a;
            src_idx_a[orig_idx] = (a->shape[orig_idx] > 1) ? dest_idx_a[j] : 0;
        }

        int flat_src_idx = get_flat_index(a, src_idx_a);
        double ref_value = a->data[flat_src_idx];
        (*out_a)->data[i] = ref_value;
    }

    free(src_idx_a);
    free(dest_idx_a);

    // Broadcast tensor b and fill in tensor out_b
    int offset_b = max_dims - b->dimensions;
    int *src_idx_b = (int *)malloc(b->dimensions * sizeof(int));
    int *dest_idx_b = (int *)malloc((*out_b)->dimensions * sizeof(int));

    for (int i = 0; i < (*out_b)->total_size; i++) {
        get_multi_dimensional_index(*out_b, i, dest_idx_b);

        for (int j = offset_b; j < max_dims; j++) {
            int orig_idx = j - offset_b;
            src_idx_b[orig_idx] = (b->shape[orig_idx] > 1) ? dest_idx_b[j] : 0;
        }
        
        int flat_src_idx = get_flat_index(b, src_idx_b);
        double ref_value = b->data[flat_src_idx];
        (*out_b)->data[i] = ref_value;
    }

    free(src_idx_b);
    free(dest_idx_b);

    // Free allocated memory
    free(broadcast_shape_a);
    free(broadcast_shape_b);
    free(state_dims_a);
    free(state_dims_b);
}

double ct_add(
    double a,
    double b,
    double *out_grad_a,
    double *out_grad_b
) {
    double out = a + b;

    if (out_grad_a) *out_grad_a = 1;
    if (out_grad_b) *out_grad_b = 1;
    
    return out;
}

double ct_sub(
    double a,
    double b,
    double *out_grad_a,
    double *out_grad_b
) { 
    double out = a - b;
    if (out_grad_a) *out_grad_a = 1;
    if (out_grad_b) *out_grad_b = -1;
    
    return out;
}

double ct_mul(
    double a,
    double b,
    double *out_grad_a,
    double *out_grad_b
) {
    double out = a * b;
    if (out_grad_a) *out_grad_a = b;
    if (out_grad_b) *out_grad_b = a;
    
    return out;
}

double ct_pow(
    double a,
    double b,
    double *out_grad_a,
    double *out_grad_b
) {
    double out = pow(a, b);
    if (out_grad_a) *out_grad_a = b * pow(a, b - 1);

    return out;
}

double ct_div(
    double a,
    double b,
    double *out_grad_a,
    double *out_grad_b
) {
    if (b == 0) {
        fprintf(stderr, "Error: Division by zero.\n");
        exit(EXIT_FAILURE);
    }

    double out = a / b;
    if (out_grad_a) *out_grad_a = 1 / b;
    if (out_grad_b) *out_grad_b = -a / (b * b);

    return out;
}

double ct_neg(
    double a,
    double *out_grad_a
) {
    double out = -a;
    if (out_grad_a) *out_grad_a = -1;

    return out;
}

double ct_exp(
    double a,
    double *out_grad_a
) {
    double out = exp(a);
    if (out_grad_a) *out_grad_a = out;
    
    return out;
}

double ct_log(
    double a,
    double *out_grad_a
) {
    double out = log(a);
    if (out_grad_a) *out_grad_a = 1.0 / a;
    
    return out;
}

double ct_tanh(
    double a,
    double *out_grad_a
) {
    double e_pos = ct_exp(a, NULL);
    double e_neg = ct_exp(-a, NULL);
    double out = (e_pos - e_neg) / (e_pos + e_neg);
    if (out_grad_a) *out_grad_a = 1 - (out * out);

    return out;
}

double ct_tan(
    double a,
    double *out_grad_a
) {
    double out = tan(a);
    double c = cos(a);
    if (out_grad_a) *out_grad_a = 1.0 / (c * c);

    return out;
}

double ct_abs(
    double a,
    double *out_grad_a
) {
    double out = a >= 0 ? a : -a;
    if (out_grad_a) *out_grad_a = a >= 0 ? 1 : -1;

    return out;
}

double ct_relu(
    double a,
    double *out_grad_a
) {
    double out = a > 0 ? a : 0;
    if (out_grad_a) *out_grad_a = a > 0 ? 1 : 0;

    return out;
}

double ct_pow2(
    double a,
    double *out_grad_a
) {
    double out = a * a;
    if (out_grad_a) *out_grad_a = 2 * a;

    return out;
}

// Function to perform matrix multiplication on 2D arrays
void ct_matrix_multiply(
    double *a,
    double *b,
    double *out,
    int n,
    int m,
    int p
) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < p; j++) {
            out[i * p + j] = 0;
            for (int k = 0; k < m; k++) {
                out[i * p + j] += a[i * m + k] * b[k * p + j];
            }
        }
    }
}

void ct_sum(
    Tensor *tensor,
    int dim,
    Tensor **out_grad_tensor,
    Tensor **out_tensor
) {
    if (dim < 0) {
        // Sum all elements in the tensor
        create_tensor_from_scalar(0.0, out_tensor);
        for (int i = 0; i < tensor->total_size; i++) {
            (*out_tensor)->data[0] += tensor->data[i];
        }
        
        // Create gradient tensor with all entries set to 1 for broadcasting
        if (out_grad_tensor) {
            create_tensor(tensor->shape, tensor->dimensions, out_grad_tensor);
            init_tensor(1.0, *out_grad_tensor);
        }

    } else {
        // Compute the new shape by removing the specified dimension and create output tensor
        if (tensor->dimensions == 2) {
            int new_shape[tensor->dimensions];
            for (int i = 0, j = 0; i < tensor->dimensions; i++) {
                if (i != dim) {
                    new_shape[j++] = tensor->shape[i];
                } else {
                    new_shape[j++] = 1;
                }
            }
            create_tensor(new_shape, tensor->dimensions, out_tensor);
        } else {
            int new_shape[tensor->dimensions - 1];
            for (int i = 0, j = 0; i < tensor->dimensions; i++) {
                if (i != dim) {
                    new_shape[j++] = tensor->shape[i];
                }
            }
            create_tensor(new_shape, tensor->dimensions - 1, out_tensor);
        }

        int outer_size = 1, inner_size = 1;
        for (int i = 0; i < dim; i++) outer_size *= tensor->shape[i];
        for (int i = dim + 1; i < tensor->dimensions; i++) inner_size *= tensor->shape[i];

        // Sum along the specified dimension
        for (int i = 0; i < outer_size; i++) {
            for (int j = 0; j < inner_size; j++) {
                double sum = 0.0;
                for (int k = 0; k < tensor->shape[dim]; k++) {
                    int idx = (i * tensor->shape[dim] * inner_size) + (k * inner_size) + j;
                    sum += tensor->data[idx];
                }
                int out_idx = i * inner_size + j;
                (*out_tensor)->data[out_idx] = sum;
            }
        }

        // Create the gradient tensor for backpropagation if requested
        if (out_grad_tensor) {
            create_tensor(tensor->shape, tensor->dimensions, out_grad_tensor);
            init_tensor(0.0, *out_grad_tensor);

            // Distribute the gradients equally across summed elements
            for (int i = 0; i < outer_size; i++) {
                for (int j = 0; j < inner_size; j++) {
                    int out_idx = i * inner_size + j;
                    for (int k = 0; k < tensor->shape[dim]; k++) {
                        int idx = (i * tensor->shape[dim] * inner_size) + (k * inner_size) + j;
                        (*out_grad_tensor)->data[idx] = 1.0;
                    }
                }
            }
        }
    }
}


void ct_softmax(
    Tensor *tensor,
    int dim,
    Tensor **out_grad_tensor,
    Tensor **out_tensor
) {
    // Create output tensor with the same shape as the input tensor
    create_tensor(tensor->shape, tensor->dimensions, out_tensor);

    // Calculate softmax along the specified dimension
    int outer_size = 1, inner_size = 1;
    for (int i = 0; i < dim; i++) outer_size *= tensor->shape[i];
    for (int i = dim + 1; i < tensor->dimensions; i++) inner_size *= tensor->shape[i];

    // Compute softmax along the specified dimension
    for (int i = 0; i < outer_size; i++) {
        for (int j = 0; j < inner_size; j++) {
            // Calculate the sum of exponentials along `dim`
            double sum_exp = 0.0;
            for (int k = 0; k < tensor->shape[dim]; k++) {
                int idx = (i * tensor->shape[dim] * inner_size) + (k * inner_size) + j;
                sum_exp += ct_exp(tensor->data[idx], NULL);
            }
            // Calculate softmax for each element along `dim`
            for (int k = 0; k < tensor->shape[dim]; k++) {
                int idx = (i * tensor->shape[dim] * inner_size) + (k * inner_size) + j;
                (*out_tensor)->data[idx] = ct_exp(tensor->data[idx], NULL) / sum_exp;
            }
        }
    }

    // If gradient tensor is requested, calculate Jacobian
    if (out_grad_tensor) {
        // Define shape for the Jacobian tensor
        int shape[tensor->dimensions + 1];
        for (int i = 0; i < tensor->dimensions + 1; i++) {
            shape[i] = (i <= dim) ? tensor->shape[i] : tensor->shape[i - 1];
        }
        create_tensor(shape, tensor->dimensions + 1, out_grad_tensor);

        int jacobian_indices[tensor->dimensions + 1];  // To hold indices in the Jacobian tensor

        for (int i = 0; i < outer_size; i++) {
            for (int j = 0; j < inner_size; j++) {
                for (int m = 0; m < tensor->shape[dim]; m++) {
                    int idx_m = (i * tensor->shape[dim] * inner_size) + (m * inner_size) + j;
                    double sm_m = (*out_tensor)->data[idx_m];
                    
                    for (int n = 0; n < tensor->shape[dim]; n++) {
                        int idx_n = (i * tensor->shape[dim] * inner_size) + (n * inner_size) + j;
                        double sm_n = (*out_tensor)->data[idx_n];

                        // Calculate gradient value
                        double grad_value = (m == n) ? sm_m * (1 - sm_m) : -sm_m * sm_n;

                        // Construct the full Jacobian index
                        int original_indices[tensor->dimensions];  // To hold original indices
                        int tmp = i * inner_size * tensor->shape[dim] + j;

                        // Calculate all original indices, both before and after dim
                        for (int d = tensor->dimensions - 1; d >= 0; d--) {
                            if (d != dim) {
                                original_indices[d] = tmp % tensor->shape[d];
                                tmp /= tensor->shape[d];
                            }
                        }

                        // Populate jacobian_indices with original indices, inserting m and n at the extra dimension
                        for (int d = 0; d < tensor->dimensions + 1; d++) {
                            if (d < dim) {
                                jacobian_indices[d] = original_indices[d];
                            } else if (d == dim) {
                                jacobian_indices[d] = m; // For the softmax dim index m
                            } else if (d == dim + 1) {
                                jacobian_indices[d] = n; // For the softmax dim index n
                            } else {
                                jacobian_indices[d] = original_indices[d - 1];
                            }
                        }

                        // Set the value in the Jacobian tensor
                        int flat_idx = get_flat_index(*out_grad_tensor, jacobian_indices);
                        (*out_grad_tensor)->data[flat_idx] = grad_value;
                    }
                }
            }
        }
    }
}

// Function to perform batch matrix multiplication
void ct_matmul(
    Tensor *a,
    Tensor *b,
    Tensor **out_result
) {
    // Dimensions of the matrices to multiply
    int a_last_dim = a->shape[a->dimensions - 1];
    int a_second_last_dim = (a->dimensions > 1) ? a->shape[a->dimensions - 2] : 1;
    int b_last_dim = b->shape[b->dimensions - 1];
    int b_second_last_dim = (b->dimensions > 1) ? b->shape[b->dimensions - 2] : 1;

    // Ensure matrix multiplication dimensions match
    if (a_last_dim != b_second_last_dim) {
        fprintf(stderr, "Error: Matrix multiplication dimensions do not align.\n");
        exit(EXIT_FAILURE);
    }

    // Calculate the number of batch dimensions for each tensor
    int a_batch_dims = a->dimensions - 2;
    int b_batch_dims = b->dimensions - 2;

    // Broadcasted batch dimensions for both tensors and preserve the last two dimensions (matrix dimensions)
    Tensor *broadcasted_a = NULL;
    Tensor *broadcasted_b = NULL;
    int preserved_dims_a[] = {a->dimensions - 1, a->dimensions - 2};
    int preserved_dims_b[] = {b->dimensions - 1, b->dimensions - 2};
    broadcast(a, b, 2, preserved_dims_a, 2, preserved_dims_b, &broadcasted_a, &broadcasted_b);

    // Create the output tensor
    int *out_shape = (int *)malloc((broadcasted_a->dimensions) * sizeof(int));
    memcpy(out_shape, broadcasted_a->shape, (broadcasted_a->dimensions - 2) * sizeof(int));
    out_shape[broadcasted_a->dimensions - 2] = a_second_last_dim;  // n from a
    out_shape[broadcasted_a->dimensions - 1] = b_last_dim;         // p from b
    create_tensor(out_shape, broadcasted_a->dimensions, out_result);

    // Iterate over the broadcasted batch dimensions
    for (int i = 0; i < calculate_total_size(broadcasted_a->shape, broadcasted_a->dimensions - 2); i++) {
        // Identify the correct slices for 'a' and 'b'
        int a_batch_idx = i * a_second_last_dim * a_last_dim;
        int b_batch_idx = i * b_second_last_dim * b_last_dim;

        double *a_slice = &broadcasted_a->data[a_batch_idx];
        double *b_slice = &broadcasted_b->data[b_batch_idx];
        double *out_slice = &(*out_result)->data[i * a_second_last_dim * b_last_dim];

        // Perform matrix multiplication for this slice
        ct_matrix_multiply(a_slice, b_slice, out_slice, a_second_last_dim, a_last_dim, b_last_dim);
    }

    // Free allocated memory
    free(out_shape);
    dispose_tensor(broadcasted_a, true);
    dispose_tensor(broadcasted_b, true);
}

// Function to transpose a tensor by swapping two dimensions
void ct_transpose_tensor(
    Tensor *tensor,
    int dim1,
    int dim2,
    Tensor **out_tensor
) {
    int *shape = tensor->shape;
    int dimensions = tensor ->dimensions;

    // Create a new tensor structure for the transpose
    create_tensor(shape, dimensions, out_tensor);
    
    // We do not need tensor_T->data because we are going to use tensor->data instead
    free((*out_tensor)->data);
    
    // Swap the specified dimensions in the shape array
    int temp_shape = (*out_tensor)->shape[dim1];
    (*out_tensor)->shape[dim1] = (*out_tensor)->shape[dim2];
    (*out_tensor)->shape[dim2] = temp_shape;
    
    // Swap the strides for the transposed dimensions
    int temp_stride = (*out_tensor)->strides[dim1];
    (*out_tensor)->strides[dim1] = (*out_tensor)->strides[dim2];
    (*out_tensor)->strides[dim2] = temp_stride;

    // Re-calculate the original strides for the transposed tensor
    calculate_strides((*out_tensor)->shape, (*out_tensor)->dimensions, (*out_tensor)->org_strides);
    
    // The data pointer is shared between input and result tensors
    (*out_tensor)->data = tensor->data;
}

// Define the operation
void operation(
    void *op,   // This is a function pointer, could point to various types of operations
    Tensor *a,
    Tensor *b,
    int dim1,
    int dim2,
    Tensor **out_grad_a,
    Tensor **out_grad_b,
    Tensor **out_result
) {
    // Check if the operation is a single-argument double operation (single tensor element-wise)
    if (op != NULL && b == NULL) {
        if ((void (*)(Tensor *, int, int, Tensor **))op == ct_transpose_tensor) {
            ct_transpose_tensor(a, dim1, dim2, out_result);
        } else if ((void (*)(Tensor *, int, Tensor **, Tensor **))op == ct_softmax) {
            ct_softmax(a, dim1, out_grad_a, out_result);
        } else if ((void (*)(Tensor *, int, Tensor **, Tensor **))op == ct_sum) {
            ct_sum(a, dim1, out_grad_a, out_result);
        } else {
            // Single-tensor operation
            double (*op_single)(double, double *) = (double (*)(double, double *))op;
            create_tensor(a->shape, a->dimensions, out_result);
            if (out_grad_a != NULL) {
                // Create gradient tensor
                create_tensor((*out_result)->shape, (*out_result)->dimensions, out_grad_a);
                for (int i = 0; i < a->total_size; i++) {
                    (*out_result)->data[i] = op_single(a->data[i], &(*out_grad_a)->data[i]);
                }
            } else {
                for (int i = 0; i < a->total_size; i++) {
                    (*out_result)->data[i] = op_single(a->data[i], NULL);
                }
            }
        }
    } else if (op != NULL && b != NULL) {
        // Check if the operation is matrix multiplication or element-wise tensor operation
        if ((void (*)(Tensor *, Tensor *, Tensor **))op == ct_matmul) {
            // Batch matrix multiplication
            ct_matmul(a, b, out_result);
        } else {
            // Two-tensor operation with broadcasting
            double (*op_double)(double, double, double *, double *) = (double (*)(double, double, double *, double *))op;
            Tensor *broadcasted_a = NULL;
            Tensor *broadcasted_b = NULL;
            broadcast(a, b, 0, NULL, 0, NULL, &broadcasted_a, &broadcasted_b);

            create_tensor(broadcasted_a->shape, broadcasted_a->dimensions, out_result);

            if (out_grad_a != NULL && out_grad_b != NULL) {
                // Both gradients required
                create_tensor((*out_result)->shape, (*out_result)->dimensions, out_grad_a);
                create_tensor((*out_result)->shape, (*out_result)->dimensions, out_grad_b);
                for (int i = 0; i < (*out_result)->total_size; i++) {
                    (*out_result)->data[i] = op_double(broadcasted_a->data[i], broadcasted_b->data[i], &(*out_grad_a)->data[i], &(*out_grad_b)->data[i]);
                }
            } else if (out_grad_a != NULL) {
                // Only out_grad_a required
                create_tensor((*out_result)->shape, (*out_result)->dimensions, out_grad_a);
                for (int i = 0; i < (*out_result)->total_size; i++) {
                    (*out_result)->data[i] = op_double(broadcasted_a->data[i], broadcasted_b->data[i], &(*out_grad_a)->data[i], NULL);
                }
            } else if (out_grad_b != NULL) {
                // Only out_grad_b required
                create_tensor((*out_result)->shape, (*out_result)->dimensions, out_grad_b);
                for (int i = 0; i < (*out_result)->total_size; i++) {
                    (*out_result)->data[i] = op_double(broadcasted_a->data[i], broadcasted_b->data[i], NULL, &(*out_grad_b)->data[i]);
                }
            } else {
                // No gradients required
                for (int i = 0; i < (*out_result)->total_size; i++) {
                    (*out_result)->data[i] = op_double(broadcasted_a->data[i], broadcasted_b->data[i], NULL, NULL);
                }
            }

            // Free broadcasted tensors
            dispose_tensor(broadcasted_a, true);
            dispose_tensor(broadcasted_b, true);
        }
    } else {
        // Handle the default error case
        fprintf(stderr, "Error: Unsupported operation or input types.\n");
        exit(EXIT_FAILURE);
    }
}

/////////////////////////////////////////////////////////////////////

enum NodeType {
    INPUT_OTHER,
    ONLY_INPUT,
    ONLY_OTHER
};

typedef struct Node {
    enum NodeType node_type;
    
    void *op;
    
    int dim1;
    int dim2;
    
    bool bypass_backward;
    bool other_parameter;
    
    Tensor *input;
    Tensor *other;
    Tensor *output;

    Tensor *target;

    Tensor *grad_input;
    Tensor *grad_other;
    Tensor *grad_output;

    Tensor **retained_grad_input;
    Tensor **retained_grad_other;

    // Pointers for linked list structure
    struct Node *next;
    struct Node *prev;
} Node;

void dispose_node(Node *node) {
    if (node == NULL) return;

    // Free all associated Tensors, checking if they exist
    bool data_dispose = node->op != ct_transpose_tensor;

    if (node->output) {
        dispose_tensor(node->output, data_dispose);
        node->output = NULL;
    }
    
    if (node->grad_input &&
        !node->bypass_backward && node->op != ct_softmax &&
        (node->node_type == ONLY_INPUT || node->node_type == INPUT_OTHER)) {
        dispose_tensor(node->grad_input, data_dispose);
        node->grad_input = NULL;
    }
    
    if (node->grad_other &&
        !node->bypass_backward && node->op != ct_softmax &&
        (node->node_type == ONLY_OTHER || node->node_type == INPUT_OTHER)) {
        dispose_tensor(node->grad_other, data_dispose);
        node->grad_other = NULL;
    }

    // Set next and prev to NULL for safety in a linked list
    if (node->prev) node->prev->next = node->next;
    if (node->next) node->next->prev = node->prev;

    // Free the node itself
    free(node);
}

void __create_node(
    enum NodeType node_type,
    void *op,
    int dim1,
    int dim2,
    bool bypass_backward,
    bool other_parameter,
    Tensor *other,
    Node *prev_node,
    Tensor **out_grad_input,
    Tensor **out_grad_other,
    Node **out_node
) {
    // Dynamically allocate memory for the new node
    *out_node = (Node *)malloc(sizeof(Node));
    
    // Initialize node with the proper values
    (*out_node)->node_type = node_type;

    (*out_node)->op = op;

    (*out_node)->dim1 = dim1;
    (*out_node)->dim2 = dim2;

    (*out_node)->bypass_backward = bypass_backward;
    (*out_node)->other_parameter = other_parameter;

    (*out_node)->input = NULL;
    (*out_node)->other = other;
    (*out_node)->output = NULL;

    (*out_node)->target = NULL;

    (*out_node)->grad_input = NULL;
    (*out_node)->grad_other = NULL;
    (*out_node)->grad_output = NULL;

    (*out_node)->retained_grad_input = out_grad_input;
    (*out_node)->retained_grad_other = out_grad_other;

    (*out_node)->next = NULL;
    (*out_node)->prev = prev_node;

    if (prev_node) prev_node->next = *out_node;
}

void create_node(
    enum NodeType node_type,
    void *op,
    int dim1,
    int dim2,
    bool bypass_backward,
    bool other_parameter,
    Tensor *other,
    Node *prev_node,
    Node **out_node
) {
    __create_node(node_type, op, dim1, dim2, bypass_backward, other_parameter, other, prev_node, NULL, NULL, out_node);
}

void create_node_retain_grad(
    enum NodeType node_type,
    void *op,
    int dim1,
    int dim2,
    bool bypass_backward,
    bool other_parameter,
    Tensor *other,
    Node *prev_node,
    Tensor **out_grad_input,
    Tensor **out_grad_other,
    Node **out_node
) {
    __create_node(node_type, op, dim1, dim2, bypass_backward, other_parameter, other, prev_node, out_grad_input, out_grad_other, out_node);
}

void feedforward(
    Tensor *X,
    Node *from_node,
    Node *to_node,
    Tensor **out_tensor
) {
    Node *node = from_node;
    
    // Set the input to the current node
    node->input = X;

    // Traverse through the remaining nodes in the graph
    while (node != to_node->next) {
        // Perform the operation and its grads and store the result in the output tensors
        if (node->bypass_backward || node->op == ct_softmax) {
            if (node->node_type == INPUT_OTHER) {
                operation(node->op, node->input, node->other, node->dim1, node->dim2, NULL, NULL, &node->output);
            } else if (node->node_type == ONLY_INPUT) {
                operation(node->op, node->input, NULL, node->dim1, node->dim2, NULL, NULL, &node->output);
            } else if (node->node_type == ONLY_OTHER) {
                operation(node->op, node->other, NULL, node->dim1, node->dim2, NULL, NULL, &node->output);
            }
        } else {
            if (node->node_type == INPUT_OTHER) {
                operation(node->op, node->input, node->other, node->dim1, node->dim2, &node->grad_input, &node->grad_other, &node->output);
            } else if (node->node_type == ONLY_INPUT) {
                operation(node->op, node->input, NULL, node->dim1, node->dim2, &node->grad_input, NULL, &node->output);
            } else if (node->node_type == ONLY_OTHER) {
                operation(node->op, node->other, NULL, node->dim1, node->dim2, &node->grad_other, NULL, &node->output);
            }
        }
        
        if (node->next) {
            if (node->node_type == ONLY_OTHER) {
                node->next->other = node->output;
                node->next->input = node->input;
            } else {
                node->next->input = node->output;
            }
        }
        
        node = node->next;
    }

    // Return the output of the last node
    *out_tensor = to_node->output;
}

void backward(Node *node) {
    // Allocate memory for the output tensor
    if (!node->grad_output) {
        create_tensor(node->output->shape, node->output->dimensions, &node->grad_output);
        init_tensor(1.0, node->grad_output);
    }

    while (node) {
        if (node->bypass_backward) {
            // Note that grad_output MUST have the same shape as grad_input and grad_other
            node->grad_input = node->grad_output;
            node->grad_other = node->grad_output;
        } else if (node->op == ct_softmax) {
            Tensor *grad_input_tmp = NULL;
            operation(ct_sub, node->output, node->target, node->dim1, node->dim2, NULL, NULL, &grad_input_tmp);
            operation(ct_mul, node->grad_output, grad_input_tmp, node->dim1, node->dim2, NULL, NULL, &node->grad_input);
            dispose_tensor(grad_input_tmp, true);
        } else if (node->op == ct_matmul) {
            Tensor *input_T = NULL;
            Tensor *other_T = NULL;
            ct_transpose_tensor(node->input, node->dim1, node->dim2, &input_T);
            ct_transpose_tensor(node->other, node->dim1, node->dim2, &other_T);
            operation(ct_matmul, node->grad_output, other_T, node->dim1, node->dim2, NULL, NULL, &node->grad_input);
            operation(ct_matmul, input_T, node->grad_output, node->dim1, node->dim2, NULL, NULL, &node->grad_other);
            dispose_tensor(input_T, false);
            dispose_tensor(other_T, false);
        } else if (node->op == ct_transpose_tensor) {
            if (node->node_type == ONLY_INPUT) {
                ct_transpose_tensor(node->grad_output, node->dim1, node->dim2, &node->grad_input);
            } else if (node->node_type == ONLY_OTHER) {
                ct_transpose_tensor(node->grad_output, node->dim1, node->dim2, &node->grad_other);
            }
        } else {
            Tensor *grad_input_tmp = NULL;
            operation(ct_mul, node->grad_input, node->grad_output, node->dim1, node->dim2, NULL, NULL, &grad_input_tmp);
            memcpy(node->grad_input->data, grad_input_tmp->data, grad_input_tmp->total_size * sizeof(double));
            dispose_tensor(grad_input_tmp, true);
            if (node->grad_other) {
                Tensor *grad_other_tmp = NULL;
                operation(ct_mul, node->grad_other, node->grad_output, node->dim1, node->dim2, NULL, NULL, &grad_other_tmp);
                memcpy(node->grad_other->data, grad_other_tmp->data, grad_other_tmp->total_size * sizeof(double));
                dispose_tensor(grad_other_tmp, true);
            }
        }

        // Let's check for broadcasting and reduce the broadcasted dims
        if (node->input && node->grad_input) {
            Tensor *tmp_grad_input = NULL;
            int max_dims = node->grad_input->dimensions;
            // Fill in the shapes starting from the leftmost dimension
            for (int i = 0; i < max_dims; i++) {
                int dim_input = (i >= max_dims - node->input->dimensions) ? node->input->shape[i - (max_dims - node->input->dimensions)] : 1;
                int dim_grad_input = (i >= max_dims - node->grad_input->dimensions) ? node->grad_input->shape[i - (max_dims - node->grad_input->dimensions)] : 1;

                // Determine the broadcasted dimension size
                if (dim_input == 1 && dim_grad_input > 1) {
                    operation(ct_sum, node->grad_input, NULL, i, -1, NULL, NULL, &tmp_grad_input);
                    dispose_tensor(node->grad_input, true);
                    deep_copy_tensor(tmp_grad_input, &node->grad_input);
                    dispose_tensor(tmp_grad_input, true);
                    max_dims = node->grad_input->dimensions;
                }
            }
        }
        if (node->other && node->grad_other) {
            Tensor *tmp_grad_other = NULL;
            int max_dims = node->grad_other->dimensions;
            // Fill in the shapes starting from the leftmost dimension
            for (int i = 0; i < max_dims; i++) {
                int dim_other = (i >= max_dims - node->other->dimensions) ? node->other->shape[i - (max_dims - node->other->dimensions)] : 1;
                int dim_grad_other = (i >= max_dims - node->grad_other->dimensions) ? node->grad_other->shape[i - (max_dims - node->grad_other->dimensions)] : 1;

                // Determine the broadcasted dimension size
                if (dim_other == 1 && dim_grad_other > 1) {
                    operation(ct_sum, node->grad_other, NULL, i, -1, NULL, NULL, &tmp_grad_other);
                    dispose_tensor(node->grad_other, true);
                    deep_copy_tensor(tmp_grad_other, &node->grad_other);
                    dispose_tensor(tmp_grad_other, true);
                    max_dims = node->grad_other->dimensions;
                }
            }
        }

        if (node->retained_grad_input) {
            *(node->retained_grad_input) = node->grad_input;
        }
        if (node->retained_grad_other) {
            *(node->retained_grad_other) = node->grad_other;
        }

        if (node->prev){
            if (node->prev->node_type == ONLY_OTHER) {
            node->prev->grad_output = node->grad_other;
            node->prev->grad_input = node->grad_input;
            } else {
                node->prev->grad_output = node->grad_input;
            }

            node->prev->target = node->target;
        }

        node = node->prev;
    }
}

// Last step, which updates parameters
void update_parameters(Node *last_node, Tensor *lr) {
    Node *node = last_node;

    while (node) {
        if (node->other_parameter) {
            Tensor *delta = NULL;
            Tensor *other_tmp = NULL;
            operation(ct_mul, lr, node->grad_other, node->dim1, node->dim2, NULL, NULL, &delta);
            operation(ct_sub, node->other, delta, node->dim1, node->dim2, NULL, NULL, &other_tmp);
            memcpy(node->other->data, other_tmp->data, other_tmp->total_size * sizeof(double));
            dispose_tensor(delta, true);
            dispose_tensor(other_tmp, true);
        }

        node = node->prev;
    }
}

// Dispose the computational graph
void dispose_graph(Node *last_node) {
    Node *node = last_node;

    dispose_tensor(node->grad_other, true);
    node->grad_other = NULL;

    while (node) {
        Node *prev_node = node->prev;
        dispose_node(node);
        node = prev_node;
    }
}

/////////////////////////////////////////////////////////////////////

typedef struct
{
    bool bias;
    int input_feature_size;
    int output_feature_size;

    Tensor *W;
    Tensor *b;

    Tensor *grad_W;
    Tensor *grad_b;

    Node *from_node;
    Node *to_node;
} LinearLayer;

LinearLayer *linearlayer(
    int input_feature_size,
    int output_feature_size,
    bool bias
) {
    LinearLayer *ll = (LinearLayer *)malloc(sizeof(LinearLayer));
    
    ll->bias = bias;
    ll->input_feature_size = input_feature_size;
    ll->output_feature_size = output_feature_size;

    create_tensor((int[]) {ll->output_feature_size, ll->input_feature_size}, 2, &ll->W);
    init_tensor_rand(1.0 / ll->input_feature_size, ll->W);

    if (bias) {
        create_tensor((int[]) {1, ll->output_feature_size}, 2, &ll->b);
        init_tensor_rand(1.0 / ll->input_feature_size, ll->b);
    }

    return ll;
}

void forward_linearlayer(
    LinearLayer *ll,
    Node *last_node,
    Tensor *X,
    Tensor **out_tensor
) {
    Node *W_T_node = NULL;
    create_node_retain_grad(ONLY_OTHER, ct_transpose_tensor, 0, 1, false, true, ll->W, last_node, NULL, &ll->grad_W, &W_T_node);
    ll->from_node = W_T_node;

    Node *X_W_T_node = NULL;
    create_node(INPUT_OTHER, ct_matmul, 0, 1, false, false, NULL, W_T_node, &X_W_T_node);

    if (ll->bias == true) {
        Node *b_node = NULL;
        create_node_retain_grad(INPUT_OTHER, ct_add, 0, 1, false, true, ll->b, X_W_T_node, NULL, &ll->grad_b, &b_node);
        ll->to_node = b_node;
    } else {
        ll->to_node = X_W_T_node;
    }
    
    feedforward(X, ll->from_node, ll->to_node, out_tensor);
}

void dispose_linearlayer(LinearLayer *ll) {
    dispose_tensor(ll->W, true); ll->W = NULL;
    if (ll->bias) dispose_tensor(ll->b, true); ll->b = NULL;
    free(ll);
}

/////////////////////////////////////////////////////////////////////

typedef struct {
    void *op;
    int dim;

    Tensor *grad_x;

    Node *from_node;
    Node *to_node;
} ActivationLayer;

ActivationLayer *activation_layer(
    void *op,
    int dim
) {
    ActivationLayer *al = (ActivationLayer *)malloc(sizeof(ActivationLayer));
    al->op = op;
    al->dim = dim;

    return al;
}

void forward_activationlayer(
    ActivationLayer *al,
    Node *last_node,
    Tensor *X,
    Tensor **out_tensor
) {
    Node *act_node = NULL;
    create_node(ONLY_INPUT, al->op, al->dim, 1, false, false, NULL, last_node, &act_node);
    al->from_node = act_node;
    al->to_node = act_node;

    feedforward(X, al->from_node, al->to_node, out_tensor);
}

void dispose_activationlayer(ActivationLayer *activationlayer) {
    free(activationlayer);
}

/////////////////////////////////////////////////////////////////////

typedef struct {
    Tensor *c;
    Tensor *two;

    Node *from_node;
    Node *to_node;
} MSELoss;

MSELoss *mseloss() {
    MSELoss *mseloss = (MSELoss *)malloc(sizeof(MSELoss));
    
    create_tensor_from_scalar(1.0, &mseloss->c);
    create_tensor_from_scalar(2.0, &mseloss->two);

    return mseloss;
}

void forward_mseloss(
    MSELoss *mseloss,
    Node *last_node,
    Tensor *X,
    Tensor *target,
    Tensor **out_tensor
) {
    Node *sub_node = NULL;
    create_node(INPUT_OTHER, ct_sub, 0, 1, false, false, target, last_node, &sub_node);
    mseloss->from_node = sub_node;

    Node *sub_pow_node = NULL;
    create_node(INPUT_OTHER, ct_pow, 0, 1, false, false, mseloss->two, sub_node, &sub_pow_node);

    Node *sub_pow_mul_node = NULL;
    mseloss->c->data[0] = 1.0 / (X->shape[0] * X->shape[1]);
    create_node(INPUT_OTHER, ct_mul, 0, 1, false, false, mseloss->c, sub_pow_node, &sub_pow_mul_node);

    Node *sub_pow_mul_sum_node = NULL;
    create_node(ONLY_INPUT, ct_sum, -1, 1, false, false, NULL, sub_pow_mul_node, &sub_pow_mul_sum_node);
    mseloss->to_node = sub_pow_mul_sum_node;

    mseloss->to_node->target = target;

    feedforward(X, mseloss->from_node, mseloss->to_node, out_tensor);
}

void dispose_mseloss(MSELoss *mseloss) {
    dispose_tensor(mseloss->two, true); mseloss->two = NULL;
    dispose_tensor(mseloss->c, true); mseloss->c = NULL;
    free(mseloss);
}

typedef struct {
    Tensor *c;

    Node *from_node;
    Node *to_node;
} NLLLoss;

NLLLoss *nllloss() {
    NLLLoss *nllloss = (NLLLoss *)malloc(sizeof(NLLLoss));
    
    create_tensor_from_scalar(1.0, &nllloss->c);

    return nllloss;
}

void forward_nllloss(
    NLLLoss *nllloss,
    Node *last_node,
    Tensor *X,
    Tensor *target,
    Tensor **out_tensor
) {
    Node *log_node = NULL;
    create_node(ONLY_INPUT, ct_log, 0, 1, true, false, NULL, last_node, &log_node);
    nllloss->from_node = log_node;

    Node *log_mul_node = NULL;
    create_node(INPUT_OTHER, ct_mul, 0, 1, true, false, target, log_node, &log_mul_node);

    Node *log_mul_sum_node = NULL;
    create_node(ONLY_INPUT, ct_sum, -1, 1, false, false, NULL, log_mul_node, &log_mul_sum_node);

    Node *log_mul_sum_mul_node = NULL;
    nllloss->c->data[0] = 1.0 / (X->shape[0]);
    create_node(INPUT_OTHER, ct_mul, 0, 1, false, false, nllloss->c, log_mul_sum_node, &log_mul_sum_mul_node);
    
    Node *log_mul_sum_mul_neg_node = NULL;
    nllloss->c->data[0] = 1.0 / (X->shape[0]);
    create_node(INPUT_OTHER, ct_neg, 0, 1, true, false, NULL, log_mul_sum_mul_node, &log_mul_sum_mul_neg_node);
    nllloss->to_node = log_mul_sum_mul_neg_node;

    nllloss->to_node->target = target;

    feedforward(X, nllloss->from_node, nllloss->to_node, out_tensor);
}

void dispose_nllloss(NLLLoss *nllloss) {
    dispose_tensor(nllloss->c, true); nllloss->c = NULL;
    free(nllloss);
}

typedef struct {
    ActivationLayer *softmax;
    NLLLoss *nllloss;

    Node *from_node;
    Node *to_node;
} CrossEntropyLoss;

CrossEntropyLoss *crossentropyloss(
    int N,
    int dim
) {
    CrossEntropyLoss *crossentropyloss = (CrossEntropyLoss *)malloc(sizeof(CrossEntropyLoss));
    crossentropyloss->softmax = activation_layer(ct_softmax, dim);
    crossentropyloss->nllloss = nllloss();

    return crossentropyloss;
}

void forward_crossentropyloss(
    CrossEntropyLoss *crossentropyloss,
    Node *last_node,
    Tensor *X,
    Tensor *target,
    Tensor **out_tensor
) {
    Tensor *y1 = NULL;
    forward_activationlayer(crossentropyloss->softmax, last_node, X, &y1);
    forward_nllloss(crossentropyloss->nllloss, crossentropyloss->softmax->to_node, y1, target, out_tensor);

    crossentropyloss->from_node = crossentropyloss->softmax->from_node;
    crossentropyloss->to_node = crossentropyloss->nllloss->to_node;
}

void dispose_crossentropyloss(CrossEntropyLoss *crossentropyloss) {
    dispose_activationlayer(crossentropyloss->softmax); crossentropyloss->softmax = NULL;
    dispose_nllloss(crossentropyloss->nllloss); crossentropyloss->nllloss = NULL;
    free(crossentropyloss);
}

/////////////////////////////////////////////////////////////////////

typedef struct {
    LinearLayer *ll1;
    ActivationLayer *al1;
    LinearLayer *ll2;
    ActivationLayer *al2;
    LinearLayer *ll3;
    ActivationLayer *al3;
    LinearLayer *ll4;
    ActivationLayer *sm;
    
    NLLLoss *loss;
} MNISTArch;

void create_mnistarch(int in_feature_size, int out_feature_size, int batch_size, MNISTArch **out_arch) {
    *out_arch = (MNISTArch *)malloc(sizeof(MNISTArch));

    int hidden_size1 = 512;
    int hidden_size2 = 256;
    int hidden_size3 = 128;

    // Initialize DNN layers
    (*out_arch)->ll1 = linearlayer(in_feature_size, hidden_size1, true);    // Input -> Hidden Layer 1
    (*out_arch)->al1 = activation_layer(ct_relu, -1);                       // Activation after Layer 1
    (*out_arch)->ll2 = linearlayer(hidden_size1, hidden_size2, true);       // Hidden Layer 1 -> Hidden Layer 2
    (*out_arch)->al2 = activation_layer(ct_relu, -1);                       // Activation after Layer 2
    (*out_arch)->ll3 = linearlayer(hidden_size2, hidden_size3, true);       // Hidden Layer 2 -> Hidden Layer 3
    (*out_arch)->al3 = activation_layer(ct_relu, -1);                       // Activation after Layer 3
    (*out_arch)->ll4 = linearlayer(hidden_size3, out_feature_size, true);   // Hidden Layer 2 -> Hidden Layer 3
    (*out_arch)->sm  = activation_layer(ct_softmax, 1);                     // Softmax

    (*out_arch)->loss = nllloss();                                          // NLLLoss
}

void dispose_mnistarch(MNISTArch *out_arch) {
    dispose_linearlayer(out_arch->ll1);
    dispose_activationlayer(out_arch->al1);
    dispose_linearlayer(out_arch->ll2);
    dispose_activationlayer(out_arch->al2);
    dispose_linearlayer(out_arch->ll3);
    dispose_activationlayer(out_arch->al3);
    dispose_linearlayer(out_arch->ll4);
    dispose_activationlayer(out_arch->sm);

    dispose_nllloss(out_arch->loss);
}

void mnistforwad(MNISTArch *arch, Tensor *X, Tensor **y_pred) {
    Tensor *y1 = NULL, *y2 = NULL, *y3 = NULL, *y4 = NULL;
    Tensor *y5 = NULL, *y6 = NULL, *y7 = NULL;

    forward_linearlayer(arch->ll1, NULL, X, &y1);
    forward_activationlayer(arch->al1, arch->ll1->to_node, y1, &y2);

    forward_linearlayer(arch->ll2, arch->al1->to_node, y2, &y3);
    forward_activationlayer(arch->al2, arch->ll2->to_node, y3, &y4);
    
    forward_linearlayer(arch->ll3, arch->al2->to_node, y4, &y5);
    forward_activationlayer(arch->al3, arch->ll3->to_node, y5, &y6);

    forward_linearlayer(arch->ll4, arch->al3->to_node, y6, &y7);
    forward_activationlayer(arch->sm, arch->ll4->to_node, y7, y_pred);
}

void mnistloss(MNISTArch *arch, Tensor *y_pred, Tensor *target, Tensor **loss) {
    forward_nllloss(arch->loss, arch->sm->to_node, y_pred, target, loss);
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
    create_mnistarch(input_size, output_size, batch_size, &arch);

    // Assume load_mnist_batch loads the MNIST batch of images and labels
    // You will need to implement this function or load the data accordingly.
    
    for (int e = 1; e <= epoch; e++) {
        printf("Epoch: %d/%d\n\n", e, epoch);
        
        double accumulated_epoch_loss = 0.0;
        for (int b = 0; b < num_batches; b++) {
            // Placeholder for batch input and labels
            Tensor *X = NULL, *Y = NULL;

            // Load a batch of data (X, Y)
            load_mnist_batch(mnist_images, mnist_labels, training_size, b, batch_size, &X, &Y);

            // Forward pass through the DNN
            Tensor *y_pred = NULL;
            mnistforwad(arch, X, &y_pred);

            // Loss calculation
            Tensor *out_loss = NULL;
            mnistloss(arch, y_pred, Y, &out_loss);
            accumulated_epoch_loss += out_loss->data[0];

            // Backpropagation of gradients
            backward(arch->loss->to_node);
            
            // Update weights
            update_parameters(arch->loss->to_node, tensor_lr);

            // Print the loss
            if ((b + 1) % num_batches_to_print == 0) {
                printf("Batch %d/%d - Loss: %.4f\n\n", b + 1, num_batches, accumulated_epoch_loss / (b + 1.0));
            }

            // Dispose computational graph and other stuff
            dispose_graph(arch->loss->to_node);
            dispose_tensor(X, true);
            dispose_tensor(Y, true);
        }
        
        printf("Total Averaged Loss: %.4f\n", accumulated_epoch_loss / (num_batches * 1.0));
        printf("-------------------\n\n");

        if (e % 3 == 0) {
            // Decay learning rate
            tensor_lr->data[0] /= 10;
            
            // Store parameters in proper files
            char filename[50];
            
            sprintf(filename, "./chckpts/checkpoint_%d_ll1_W.txt", e);
            store_tensor(filename, arch->ll1->W, 16);
            sprintf(filename, "./chckpts/checkpoint_%d_ll1_b.txt", e);
            store_tensor(filename, arch->ll1->b, 16);
            sprintf(filename, "./chckpts/checkpoint_%d_ll2_b.txt", e);
            store_tensor(filename, arch->ll2->b, 16);
            sprintf(filename, "./chckpts/checkpoint_%d_ll2_W.txt", e);
            store_tensor(filename, arch->ll2->W, 16);
            sprintf(filename, "./chckpts/checkpoint_%d_ll3_b.txt", e);
            store_tensor(filename, arch->ll3->b, 16);
            sprintf(filename, "./chckpts/checkpoint_%d_ll3_W.txt", e);
            store_tensor(filename, arch->ll3->W, 16);
            sprintf(filename, "./chckpts/checkpoint_%d_ll4_b.txt", e);
            store_tensor(filename, arch->ll4->b, 16);
            sprintf(filename, "./chckpts/checkpoint_%d_ll4_W.txt", e);
            store_tensor(filename, arch->ll4->W, 16);
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
    create_mnistarch(input_size, output_size, batch_size, &arch);

    // Initialize DNN from the stored parameters
    load_tensor("./chckpts/checkpoint_9_ll1_W.txt", arch->ll1->W);
    load_tensor("./chckpts/checkpoint_9_ll1_b.txt", arch->ll1->b);
    load_tensor("./chckpts/checkpoint_9_ll2_W.txt", arch->ll2->W);
    load_tensor("./chckpts/checkpoint_9_ll2_b.txt", arch->ll2->b);
    load_tensor("./chckpts/checkpoint_9_ll3_W.txt", arch->ll3->W);
    load_tensor("./chckpts/checkpoint_9_ll3_b.txt", arch->ll3->b);
    load_tensor("./chckpts/checkpoint_9_ll4_W.txt", arch->ll4->W);
    load_tensor("./chckpts/checkpoint_9_ll4_b.txt", arch->ll4->b);

    double accumulated_epoch_loss = 0.0;
    for (int b = 0; b < num_batches; b++) {
        // Placeholder for batch input and labels
        Tensor *X = NULL, *Y = NULL;

        // Load a batch of data (X, Y)
        load_mnist_batch(mnist_images, mnist_labels, eval_size, b, batch_size, &X, &Y);

        // Forward pass through the DNN
        Tensor *y_pred = NULL;
        mnistforwad(arch, X, &y_pred);

        // Loss calculation
        Tensor *out_loss = NULL;
        mnistloss(arch, y_pred, Y, &out_loss);
        accumulated_epoch_loss += out_loss->data[0];

        // Dispose computational graph and other stuff
        dispose_graph(arch->loss->to_node);
        dispose_tensor(X, true);
        dispose_tensor(Y, true);
    }
    
    printf("Total Averaged Loss: %.4f\n", accumulated_epoch_loss / (num_batches * 1.0));
    
    dispose_mnistarch(arch);
    dispose_mnist_dataset(dataset_size, image_size, mnist_images, mnist_labels);
}

/////////////////////////////////////////////////////////////////////

#define SIMPLE_TEST_DATASET_SIZE 64000
#define SIMPLE_TEST_FEATURE_SIZE 1

typedef struct {
    LinearLayer *ll1;
    LinearLayer *ll2;
    ActivationLayer *al;
    
    MSELoss *loss;
} SimpleArch;

void simplearch(int input_size, int output_size, int batch_size, SimpleArch **out_arch) {
    *out_arch = (SimpleArch *)malloc(sizeof(SimpleArch));

    // Initialize DNN layers
    (*out_arch)->ll1 = linearlayer(input_size, 1, true);
    (*out_arch)->ll2 = linearlayer(1, output_size, true);
    (*out_arch)->al  = activation_layer(ct_pow2, -1);

    (*out_arch)->loss = mseloss();
}

void simpleforwad(SimpleArch *arch, Tensor *X, Tensor **y_pred) {
    Tensor *y1 = NULL, *y2 = NULL;

    forward_linearlayer(arch->ll1, NULL, X, &y1);
    forward_linearlayer(arch->ll2, arch->ll1->to_node, y1, &y2);
    forward_activationlayer(arch->al, arch->ll2->to_node, y2, y_pred);
}

void simpleloss(SimpleArch *arch, Tensor *y_pred, Tensor *target, Tensor **loss) {
    forward_mseloss(arch->loss, arch->al->to_node, y_pred, target, loss);
}

void shuffle_data(
    Tensor *X, 
    Tensor *Y, 
    int dataset_size, 
    int feature_size
) {
    // Fisher-Yates shuffle algorithm
    for (int i = dataset_size - 1; i > 0; i--) {
        // Generate a random index j such that 0 <= j <= i
        int j = rand() % (i + 1);

        // Swap row i with row j in X
        for (int k = 0; k < feature_size; k++) {
            double temp = get_element(X, i, k);
            set_element(X, get_element(X, j, k), i, k);
            set_element(X, temp, j, k);
        }

        // Swap the corresponding row i with row j in Y
        for (int k = 0; k < feature_size; k++) {
            double temp = get_element(Y, i, k);
            set_element(Y, get_element(Y, j, k), i, k);
            set_element(Y, temp, j, k);
        }
    }
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
    
    // Shuffle the data of out_X and keep out_Y in sync
    shuffle_data(*out_X, *out_Y, dataset_size, feature_size);
}

// Function to load a batch of simple_test data into tensors X and Y
void load_simple_test_batch(
    Tensor *X_dataset,
    Tensor *Y_dataset,
    int batch_idx,
    int batch_size,
    int feature_size,
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
}

void simple_test() {
    int dataset_size = SIMPLE_TEST_DATASET_SIZE;
    int feature_size = SIMPLE_TEST_FEATURE_SIZE;

    int batch_size = 64;
    int num_batches = 1000;
    int epoch = 5;
    double lr = 0.001;

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

        for (int j = 0; j < num_batches + 1; j++) {
            load_simple_test_batch(X_dataset, Y_dataset, j, batch_size, feature_size, X, Y);

            // Forward pass through the DNN
            Tensor *y_pred = NULL;
            simpleforwad(arch, X, &y_pred);

            // Loss calculation
            Tensor *out_loss = NULL;
            simpleloss(arch, y_pred, Y, &out_loss);

            // Backpropagation of gradients
            backward(arch->loss->to_node);
            
            // Update weights
            update_parameters(arch->loss->to_node, tensor_lr);

            // Every 100 batches, print the loss (MSE)
            if (j % 100 == 0) {
                printf("Batch %d/%d:\n", j, num_batches);
                printf("W1:\n");
                print_info(arch->ll1->W);
                printf("W2:\n");
                print_info(arch->ll2->W);
                printf("Loss:\n");
                print_info_with_precision(out_loss, 16);  // Print the loss tensor
                printf("-------------------\n");
            }

            // Dispose computational graph
            dispose_graph(arch->loss->to_node);
        }
    }
}

/////////////////////////////////////////////////////////////////////

void simple_test2() {
    int shape[2] = {3, 3};
    Tensor *x = NULL;
    create_tensor(shape, 2, &x);
    Tensor *y = NULL;
    create_tensor(shape, 2, &y);

    set_element(x, 1, 0, 0);
    set_element(x, 3, 0, 1);
    set_element(x, 5, 0, 2);
    set_element(x, 3, 1, 0);
    set_element(x, 2, 1, 1);
    set_element(x, 10, 1, 2);
    set_element(x, 7, 2, 0);
    set_element(x, 4, 2, 1);
    set_element(x, 7, 2, 2);
    
    set_element(y, 1, 0, 0);
    set_element(y, 0, 0, 1);
    set_element(y, 0, 0, 2);
    set_element(y, 0, 1, 0);
    set_element(y, 0, 1, 1);
    set_element(y, 1, 1, 2);
    set_element(y, 1, 2, 0);
    set_element(y, 0, 2, 1);
    set_element(y, 0, 2, 2);

    CrossEntropyLoss *cel = crossentropyloss(3, 1);

    Tensor *loss = NULL;
    forward_crossentropyloss(cel, NULL, x, y, &loss);

    print_info(x);
    print_info(y);
    print_info(loss);

    backward(cel->to_node);

    print_info(cel->softmax->to_node->grad_output);
    print_info(cel->softmax->to_node->grad_input);
}

/////////////////////////////////////////////////////////////////////

// Function to setup the whole application
void setup_application(
    int default_seed
) {
    // Seed value for random number generation
    unsigned int seed = (default_seed < 0) ? time(NULL) * time(NULL) : default_seed;
    printf("Application Seed: %u\n\n", seed);

    // Seed the random number generator, usually done once per program run
    srand(seed);
}

int main(
    int argc,
    char *argv[]
) {
    setup_application(42);

    mnist_train("./mnist_train_small.csv");
    mnist_eval("./mnist_test.csv");

    // simple_test();
    // simple_test2();

    return 0;
}