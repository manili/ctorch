#include <math.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>
#include <stdbool.h>

/////////////////////////////////////////////////////////////////////

// Structure to hold an n-dimensional tensor with index mapping
typedef struct Tensor {
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
void __create_tensor(
    int *shape,
    int dimensions,
    bool allocate_data,
    bool calc_strides,
    Tensor **out_tensor
) {
    int total_size = calculate_total_size(shape, dimensions);

    // Allocate memory and initialization
    *out_tensor = (Tensor *)malloc(sizeof(Tensor));
    (*out_tensor)->shape = (int *)malloc(dimensions * sizeof(int));
    (*out_tensor)->strides = (int *)malloc(dimensions * sizeof(int));
    (*out_tensor)->org_strides = (int *)malloc(dimensions * sizeof(int));
    (*out_tensor)->data = (allocate_data) ? (double *)malloc(total_size * sizeof(double)) : NULL;
    
    (*out_tensor)->dimensions = dimensions;
    (*out_tensor)->total_size = total_size;
    
    memcpy((*out_tensor)->shape, shape, dimensions * sizeof(int));

    if (calc_strides) {
        calculate_strides(shape, dimensions, (*out_tensor)->strides);
        calculate_strides(shape, dimensions, (*out_tensor)->org_strides);
    }
}

void create_tensor(
    int *shape,
    int dimensions,
    Tensor **out_tensor
) {
    __create_tensor(shape, dimensions, true, true, out_tensor);
}

void create_tensor_without_data(
    int *shape,
    int dimensions,
    Tensor **out_tensor
) {
    __create_tensor(shape, dimensions, false, true, out_tensor);
}

void create_tensor_like(
    Tensor *tensor,
    Tensor **out_tensor
) {
    __create_tensor(tensor->shape, tensor->dimensions, true, false, out_tensor);

    // When creating a tensor from another one, strides must be preserved
    memcpy((*out_tensor)->strides, tensor->strides, tensor->dimensions * sizeof(int));
    memcpy((*out_tensor)->org_strides, tensor->org_strides, tensor->dimensions * sizeof(int));
}

void create_tensor_like_without_data(
    Tensor *tensor,
    Tensor **out_tensor
) {
    __create_tensor(tensor->shape, tensor->dimensions, false, false, out_tensor);

    // When creating a tensor from another one, strides must be preserved
    memcpy((*out_tensor)->strides, tensor->strides, tensor->dimensions * sizeof(int));
    memcpy((*out_tensor)->org_strides, tensor->org_strides, tensor->dimensions * sizeof(int));
}

void deep_copy_tensor(
    Tensor *tensor,
    Tensor **out_tensor
) {
    create_tensor_like(tensor, out_tensor);
    memcpy((*out_tensor)->data, tensor->data, tensor->total_size * sizeof(double));
}

void create_tensor_from_scalar(
    double value,
    Tensor **out_tensor
) {
    int tensor_shape[] = {1, 1};
    create_tensor(tensor_shape, 2, out_tensor);
    (*out_tensor)->data[0] = value;
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

/////////////////////////////////////////////////////////////////////

bool tensor_broadcast(Tensor *, Tensor *, int, int *, int, int *, Tensor **, Tensor **);
void tensor_sum      (Tensor *, int, bool, Tensor **);
void tensor_reduce   (Tensor *, Tensor *, Tensor **);
void tensor_transpose(Tensor *, int, int, bool, Tensor **);
void tensor_matmul   (Tensor *, Tensor *, Tensor **);
void tensor_softmax  (Tensor *, int, Tensor **);
void tensor_neg      (Tensor *, Tensor **);
void tensor_log      (Tensor *, Tensor **);
void tensor_tan      (Tensor *, Tensor **);
void tensor_tanh     (Tensor *, Tensor **);
void tensor_exp      (Tensor *, Tensor **);
void tensor_relu     (Tensor *, Tensor **);
void tensor_abs      (Tensor *, Tensor **);
void tensor_add      (Tensor *, Tensor *, Tensor **);
void tensor_sub      (Tensor *, Tensor *, Tensor **);
void tensor_mul      (Tensor *, Tensor *, Tensor **);
void tensor_div      (Tensor *, Tensor *, Tensor **);
void tensor_pow      (Tensor *, Tensor *, Tensor **);
void tensor_view     (Tensor *, Tensor *, bool, Tensor **);

void grad_tensor_sum      (Tensor *, Tensor *, Tensor **);
void grad_tensor_transpose(Tensor *, int, int, Tensor **);
void grad_tensor_matmul   (Tensor *, Tensor *, Tensor *, Tensor **, Tensor **);
void grad_tensor_softmax  (Tensor *, Tensor *, Tensor *, int, Tensor **);
void grad_tensor_neg      (Tensor *, Tensor **);
void grad_tensor_log      (Tensor *, Tensor *, Tensor **);
void grad_tensor_tan      (Tensor *, Tensor *, Tensor **);
void grad_tensor_tanh     (Tensor *, Tensor *, Tensor **);
void grad_tensor_exp      (Tensor *, Tensor *, Tensor **);
void grad_tensor_relu     (Tensor *, Tensor *, Tensor **);
void grad_tensor_abs      (Tensor *, Tensor *, Tensor **);
void grad_tensor_add      (Tensor *, Tensor *, Tensor *, Tensor **, Tensor **);
void grad_tensor_sub      (Tensor *, Tensor *, Tensor *, Tensor **, Tensor **);
void grad_tensor_mul      (Tensor *, Tensor *, Tensor *, Tensor **, Tensor **);
void grad_tensor_div      (Tensor *, Tensor *, Tensor *, Tensor **, Tensor **);
void grad_tensor_pow      (Tensor *, Tensor *, Tensor *, Tensor *, Tensor **, Tensor **);
void grad_tensor_view     (Tensor *, Tensor *, Tensor **);

///////////////////////////////

// Function to broadcast tensors so that they would align each other
bool tensor_broadcast(
    Tensor *a,
    Tensor *b,
    int num_preserved_dims_a,
    int *preserved_dims_a,
    int num_preserved_dims_b,
    int *preserved_dims_b,
    Tensor **out_a,
    Tensor **out_b
) {
    bool need_broadcasting = false;

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
        if ((state_dims_a[i] == false || state_dims_b[i] == false) && dim_a != dim_b) {
            need_broadcasting = true;
        }

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

    if (need_broadcasting == false) {
        // Free allocated memory
        free(broadcast_shape_a);
        free(broadcast_shape_b);
        free(state_dims_a);
        free(state_dims_b);
        deep_copy_tensor(a, out_a);
        deep_copy_tensor(b, out_b);
        return false;
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

    return true;
}

///////////////////////////////

void tensor_sum(
    Tensor *a,
    int dim,
    bool keepdim,
    Tensor **out_tensor
) {
    if (dim < 0) {
        // Sum all elements in the tensor
        create_tensor_from_scalar(0.0, out_tensor);
        for (int i = 0; i < a->total_size; i++) {
            (*out_tensor)->data[0] += a->data[i];
        }
    } else {
        // Determine the output shape based on `keepdim`
        int out_dims = (a->dimensions == 2 || keepdim) ? a->dimensions : a->dimensions - 1;
        int new_shape[out_dims];

        for (int i = 0, j = 0; i < a->dimensions; i++) {
            if (i == dim) {
                if (a->dimensions == 2 || keepdim) {
                    new_shape[j++] = 1; // Keep the dimension with size 1
                }
            } else {
                new_shape[j++] = a->shape[i];
            }
        }
        create_tensor(new_shape, out_dims, out_tensor);

        // Calculate outer and inner sizes
        int outer_size = 1, inner_size = 1;
        for (int i = 0; i < dim; i++) outer_size *= a->shape[i];
        for (int i = dim + 1; i < a->dimensions; i++) inner_size *= a->shape[i];

        // Sum along the specified dimension
        for (int i = 0; i < outer_size; i++) {
            for (int j = 0; j < inner_size; j++) {
                double sum = 0.0;
                for (int k = 0; k < a->shape[dim]; k++) {
                    int idx = (i * a->shape[dim] * inner_size) + (k * inner_size) + j;
                    sum += a->data[idx];
                }
                int out_idx = i * inner_size + j;
                (*out_tensor)->data[out_idx] = sum;
            }
        }
    }
}
void grad_tensor_sum(
    Tensor *a,
    Tensor *grad,
    Tensor **out_grad_a
) {
    if (out_grad_a) {
        Tensor *broadcasted_a = NULL;
        Tensor *broadcasted_grad = NULL;
        tensor_broadcast(a, grad, 0, NULL, 0, NULL, &broadcasted_a, &broadcasted_grad);
        create_tensor(broadcasted_a->shape, broadcasted_a->dimensions, out_grad_a);

        for (int i = 0; i < broadcasted_a->total_size; i++) {
            (*out_grad_a)->data[i] = broadcasted_grad->data[i];
        }

        dispose_tensor(broadcasted_a, true);
        dispose_tensor(broadcasted_grad, true);
    }
}

///////////////////////////////

void tensor_reduce(
    Tensor *source_tensor,
    Tensor *target_tensor,
    Tensor **out_tensor
) {
    deep_copy_tensor(source_tensor, out_tensor);
    int max_dims = (*out_tensor)->dimensions;

    // Fill in the shapes starting from the leftmost dimension
    for (int i = 0; i < max_dims; i++) {
        int dim_out_size = (*out_tensor)->shape[i];
        int dim_dst_size = (i >= max_dims - target_tensor->dimensions) ? target_tensor->shape[i - (max_dims - target_tensor->dimensions)] : 1;

        // Determine the broadcasted dimension size
        if (dim_dst_size == 1 && dim_out_size > 1) {
            Tensor *tmp_grad_input = NULL;
            if (i >= max_dims - target_tensor->dimensions) {
                tensor_sum(*out_tensor, i, true, &tmp_grad_input);
            } else {
                tensor_sum(*out_tensor, i, false, &tmp_grad_input);
            }
            dispose_tensor(*out_tensor, true);
            deep_copy_tensor(tmp_grad_input, out_tensor);
            dispose_tensor(tmp_grad_input, true);
            max_dims = (*out_tensor)->dimensions;
        } else if (dim_dst_size != dim_out_size && dim_dst_size > 1 && dim_out_size > 1) {
            // Handle the default error case
            fprintf(stderr, "Error: Can not reduce source tensor to the target tensor.\n");
            exit(EXIT_FAILURE);
        }
    }
}

///////////////////////////////

// Function to transpose a tensor by swapping two dimensions
void tensor_transpose(
    Tensor *a,
    int dim1,
    int dim2,
    bool clone_data,
    Tensor **out_tensor
) {
    if (clone_data) {
        // Create a new tensor structure with transposed shape
        create_tensor(a->shape, a->dimensions, out_tensor);

        // Swap the dimensions in the shape array
        int temp_shape = (*out_tensor)->shape[dim1];
        (*out_tensor)->shape[dim1] = (*out_tensor)->shape[dim2];
        (*out_tensor)->shape[dim2] = temp_shape;

        // Re-calculate strides for a contiguous layout
        calculate_strides((*out_tensor)->shape, (*out_tensor)->dimensions, (*out_tensor)->strides);
        calculate_strides((*out_tensor)->shape, (*out_tensor)->dimensions, (*out_tensor)->org_strides);

        // Copy data from the input tensor to the output tensor with transposed indices
        for (int i = 0; i < a->total_size; i++) {
            int indices[a->dimensions];
            get_multi_dimensional_index(a, i, indices);

            // Swap the indices for the transposed dimensions
            int temp_index = indices[dim1];
            indices[dim1] = indices[dim2];
            indices[dim2] = temp_index;

            // Calculate the flat index for the output tensor
            int flat_index = get_flat_index(*out_tensor, indices);

            // Copy the data
            (*out_tensor)->data[flat_index] = a->data[i];
        }
    } else {
        // The data pointer is shared between input and result tensors
        create_tensor_without_data(a->shape, a->dimensions, out_tensor);
        (*out_tensor)->data = a->data;

        // Swap the dimensions in the shape array
        int temp_shape = (*out_tensor)->shape[dim1];
        (*out_tensor)->shape[dim1] = (*out_tensor)->shape[dim2];
        (*out_tensor)->shape[dim2] = temp_shape;

        // Swap the strides for the transposed dimensions
        int temp_stride = (*out_tensor)->strides[dim1];
        (*out_tensor)->strides[dim1] = (*out_tensor)->strides[dim2];
        (*out_tensor)->strides[dim2] = temp_stride;

        // Re-calculate the original strides for the transposed tensor
        calculate_strides((*out_tensor)->shape, (*out_tensor)->dimensions, (*out_tensor)->org_strides);
    }
}
void grad_tensor_transpose(
    Tensor *grad,
    int dim1,
    int dim2,
    Tensor **out_grad_a
) {
    tensor_transpose(grad, dim1, dim2, true, out_grad_a);
}

///////////////////////////////

// Helper function to perform matrix multiplication on 2D arrays with strides
void __matrix_multiply_strided(
    double *a, int *a_strides, // Data pointer and strides for A
    double *b, int *b_strides, // Data pointer and strides for B
    double *out, int *out_strides, // Data pointer and strides for the output
    int n, int m, int p // Dimensions: n x m x p
) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < p; j++) {
            // Compute the address of the current output element using strides
            double *out_ptr = out + i * out_strides[0] + j * out_strides[1];
            *out_ptr = 0;

            for (int k = 0; k < m; k++) {
                // Compute the addresses for the current A and B elements using strides
                double *a_ptr = a + i * a_strides[0] + k * a_strides[1];
                double *b_ptr = b + k * b_strides[0] + j * b_strides[1];

                // Perform the multiplication and add to the output
                *out_ptr += (*a_ptr) * (*b_ptr);
            }
        }
    }
}
// Function to perform batch matrix multiplication
void tensor_matmul(
    Tensor *a,
    Tensor *b,
    Tensor **out_tensor
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
    tensor_broadcast(a, b, 2, preserved_dims_a, 2, preserved_dims_b, &broadcasted_a, &broadcasted_b);

    // Create the output tensor
    int *out_shape = (int *)malloc((broadcasted_a->dimensions) * sizeof(int));
    memcpy(out_shape, broadcasted_a->shape, (broadcasted_a->dimensions - 2) * sizeof(int));
    out_shape[broadcasted_a->dimensions - 2] = a_second_last_dim;  // n from a
    out_shape[broadcasted_a->dimensions - 1] = b_last_dim;         // p from b
    create_tensor(out_shape, broadcasted_a->dimensions, out_tensor);

    // Get strides for each tensor
    int *a_strides = broadcasted_a->strides;
    int *b_strides = broadcasted_b->strides;
    int *out_strides = (*out_tensor)->strides;

    // Iterate over the broadcasted batch dimensions
    for (int i = 0; i < calculate_total_size(broadcasted_a->shape, broadcasted_a->dimensions - 2); i++) {
        // Identify the correct slices for 'a' and 'b'
        int a_batch_idx = i * a_second_last_dim * a_last_dim;
        int b_batch_idx = i * b_second_last_dim * b_last_dim;
        int out_batch_idx = i * a_second_last_dim * b_last_dim;

        double *a_slice = &broadcasted_a->data[a_batch_idx];
        double *b_slice = &broadcasted_b->data[b_batch_idx];
        double *out_slice = &(*out_tensor)->data[out_batch_idx];

        // Perform matrix multiplication for this slice
        __matrix_multiply_strided(
            a_slice, a_strides + broadcasted_a->dimensions - 2,
            b_slice, b_strides + broadcasted_b->dimensions - 2,
            out_slice, out_strides + (*out_tensor)->dimensions - 2,
            a_second_last_dim, a_last_dim, b_last_dim
        );
    }

    // Free allocated memory
    free(out_shape);
    dispose_tensor(broadcasted_a, true);
    dispose_tensor(broadcasted_b, true);
}
void grad_tensor_matmul(
    Tensor *a,
    Tensor *b,
    Tensor *grad,
    Tensor **out_grad_a,
    Tensor **out_grad_b
) {
    Tensor *tmp_out_grad_a = NULL, *tmp_out_grad_b = NULL;
    
    if (out_grad_a) {
        Tensor *b_T = NULL;
        int dim1 = b->dimensions - 2, dim2 = b->dimensions - 1;
        tensor_transpose(b, dim1, dim2, false, &b_T);
        tensor_matmul(grad, b_T, &tmp_out_grad_a);
        dispose_tensor(b_T, false);
        tensor_reduce(tmp_out_grad_a, a, out_grad_a);
        dispose_tensor(tmp_out_grad_a, true);
    }

    if (out_grad_b) {
        Tensor *a_T = NULL;
        int dim1 = a->dimensions - 2, dim2 = a->dimensions - 1;
        tensor_transpose(a, dim1, dim2, false, &a_T);
        tensor_matmul(a_T, grad, &tmp_out_grad_b);
        dispose_tensor(a_T, false);
        tensor_reduce(tmp_out_grad_b, b, out_grad_b);
        dispose_tensor(tmp_out_grad_b, true);
    }
}

///////////////////////////////

void tensor_softmax(
    Tensor *a,
    int dim,
    Tensor **out_tensor
) {
    // Create output tensor with the same shape as the input
    create_tensor(a->shape, a->dimensions, out_tensor);

    // Calculate softmax along the specified dimension
    int outer_size = 1, inner_size = 1;
    for (int i = 0; i < dim; i++) outer_size *= a->shape[i];
    for (int i = dim + 1; i < a->dimensions; i++) inner_size *= a->shape[i];

    // Compute softmax along the specified dimension
    for (int i = 0; i < outer_size; i++) {
        for (int j = 0; j < inner_size; j++) {
            // Calculate the sum of exponentials along `dim`
            double sum_exp = 0.0;
            for (int k = 0; k < a->shape[dim]; k++) {
                int idx = (i * a->shape[dim] * inner_size) + (k * inner_size) + j;
                sum_exp += exp(a->data[idx]);
            }
            // Calculate softmax for each element along `dim`
            for (int k = 0; k < a->shape[dim]; k++) {
                int idx = (i * a->shape[dim] * inner_size) + (k * inner_size) + j;
                (*out_tensor)->data[idx] = exp(a->data[idx]) / sum_exp;
            }
        }
    }
}
void grad_tensor_softmax(
    Tensor *a,
    Tensor *out,
    Tensor *grad,
    int dim,
    Tensor **out_grad_a
) {
    // If gradient tensor is requested, calculate Jacobian
    if (out_grad_a) {
        Tensor *J = NULL;

        // Calculate softmax along the specified dimension
        int outer_size = 1, inner_size = 1;
        for (int i = 0; i < dim; i++) outer_size *= a->shape[i];
        for (int i = dim + 1; i < a->dimensions; i++) inner_size *= a->shape[i];

        // Define shape for the Jacobian tensor
        int shape[a->dimensions + 1];
        for (int i = 0; i < a->dimensions + 1; i++) {
            shape[i] = (i <= dim) ? a->shape[i] : a->shape[i - 1];
        }
        create_tensor(shape, a->dimensions + 1, &J);

        int jacobian_indices[a->dimensions + 1];  // To hold indices in the Jacobian tensor

        for (int i = 0; i < outer_size; i++) {
            for (int j = 0; j < inner_size; j++) {
                for (int m = 0; m < a->shape[dim]; m++) {
                    int idx_m = (i * a->shape[dim] * inner_size) + (m * inner_size) + j;
                    double sm_m = out->data[idx_m];
                    
                    for (int n = 0; n < a->shape[dim]; n++) {
                        int idx_n = (i * a->shape[dim] * inner_size) + (n * inner_size) + j;
                        double sm_n = out->data[idx_n];

                        // Calculate gradient value
                        double grad_value = (m == n) ? sm_m * (1 - sm_m) : -sm_m * sm_n;

                        // Construct the full Jacobian index
                        int original_indices[a->dimensions];  // To hold original indices
                        int tmp = i * inner_size * a->shape[dim] + j;

                        // Calculate all original indices, both before and after dim
                        for (int d = a->dimensions - 1; d >= 0; d--) {
                            if (d != dim) {
                                original_indices[d] = tmp % a->shape[d];
                                tmp /= a->shape[d];
                            }
                        }

                        // Populate jacobian_indices with original indices, inserting m and n at the extra dimension
                        for (int d = 0; d < a->dimensions + 1; d++) {
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
                        int flat_idx = get_flat_index(J, jacobian_indices);
                        J->data[flat_idx] = grad_value;
                    }
                }
            }
        }

        tensor_matmul(grad, J, out_grad_a);
    }
}

///////////////////////////////

void tensor_neg(
    Tensor *a,
    Tensor **out_tensor
) {
    create_tensor(a->shape, a->dimensions, out_tensor);

    for (int i = 0; i < a->total_size; i++) {
        (*out_tensor)->data[i] = -1 * a->data[i];
    }
}
void grad_tensor_neg(
    Tensor *grad,
    Tensor **out_grad_a
) {
    if (out_grad_a) {
        create_tensor(grad->shape, grad->dimensions, out_grad_a);

        for (int i = 0; i < grad->total_size; i++) {
            (*out_grad_a)->data[i] = -1 * grad->data[i];
        }
    }
}

///////////////////////////////

void tensor_log(
    Tensor *a,
    Tensor **out_tensor
) {
    create_tensor(a->shape, a->dimensions, out_tensor);

    for (int i = 0; i < a->total_size; i++) {
        (*out_tensor)->data[i] = log(a->data[i]);
    }
}
void grad_tensor_log(
    Tensor *a,
    Tensor *grad,
    Tensor **out_grad_a
) {
    if (out_grad_a) {
        create_tensor(a->shape, a->dimensions, out_grad_a);

        for (int i = 0; i < a->total_size; i++) {
            (*out_grad_a)->data[i] = grad->data[i] / a->data[i];
        }
    }
}

///////////////////////////////

void tensor_tan(
    Tensor *a,
    Tensor **out_tensor
) {
    create_tensor(a->shape, a->dimensions, out_tensor);

    for (int i = 0; i < a->total_size; i++) {
        (*out_tensor)->data[i] = tan(a->data[i]);
    }
}
void grad_tensor_tan(
    Tensor *a,
    Tensor *grad,
    Tensor **out_grad_a
) {
    if (out_grad_a) {
        create_tensor(a->shape, a->dimensions, out_grad_a);

        for (int i = 0; i < a->total_size; i++) {
            double c = cos(a->data[i]);
            (*out_grad_a)->data[i] = grad->data[i] / (c * c);
        }
    }
}

///////////////////////////////

void tensor_tanh(
    Tensor *a,
    Tensor **out_tensor
) {
    create_tensor(a->shape, a->dimensions, out_tensor);

    for (int i = 0; i < a->total_size; i++) {
        (*out_tensor)->data[i] = tanh(a->data[i]);
    }
}
void grad_tensor_tanh(
    Tensor *out,
    Tensor *grad,
    Tensor **out_grad_a
) {
    if (out_grad_a) {
        create_tensor(out->shape, out->dimensions, out_grad_a);

        for (int i = 0; i < out->total_size; i++) {
            double out_pow_2 = out->data[i] * out->data[i];
            (*out_grad_a)->data[i] = grad->data[i] * (1 - out_pow_2);
        }
    }
}

///////////////////////////////

void tensor_exp(
    Tensor *a,
    Tensor **out_tensor
) {
    create_tensor(a->shape, a->dimensions, out_tensor);

    for (int i = 0; i < a->total_size; i++) {
        (*out_tensor)->data[i] = exp(a->data[i]);
    }
}
void grad_tensor_exp(
    Tensor *out,
    Tensor *grad,
    Tensor **out_grad_a
) {
    if (out_grad_a) {
        create_tensor(out->shape, out->dimensions, out_grad_a);

        for (int i = 0; i < out->total_size; i++) {
            (*out_grad_a)->data[i] = grad->data[i] * out->data[i];
        }
    }
}

///////////////////////////////

void tensor_relu(
    Tensor *a,
    Tensor **out_tensor
) {
    create_tensor(a->shape, a->dimensions, out_tensor);

    for (int i = 0; i < a->total_size; i++) {
        (*out_tensor)->data[i] = (a->data[i] > 0) ? a->data[i] : 0;
    }
}
void grad_tensor_relu(
    Tensor *a,
    Tensor *grad,
    Tensor **out_grad_a
) {
    if (out_grad_a) {
        create_tensor(a->shape, a->dimensions, out_grad_a);

        for (int i = 0; i < a->total_size; i++) {
            (*out_grad_a)->data[i] = (a->data[i] > 0) ? grad->data[i] : 0;
        }
    }
}

///////////////////////////////

void tensor_abs(
    Tensor *a,
    Tensor **out_tensor
) {
    create_tensor(a->shape, a->dimensions, out_tensor);

    for (int i = 0; i < a->total_size; i++) {
        (*out_tensor)->data[i] = (a->data[i] > 0) ? a->data[i] : -1 * a->data[i];
    }
}
void grad_tensor_abs(
    Tensor *a,
    Tensor *grad,
    Tensor **out_grad_a
) {
    if (out_grad_a) {
        create_tensor(a->shape, a->dimensions, out_grad_a);

        for (int i = 0; i < a->total_size; i++) {
            (*out_grad_a)->data[i] = (a->data[i] > 0) ? grad->data[i] : -1 * grad->data[i];
        }
    }
}

///////////////////////////////

void tensor_add(
    Tensor *a,
    Tensor *b,
    Tensor **out_tensor
) {
    Tensor *broadcasted_a = NULL;
    Tensor *broadcasted_b = NULL;
    tensor_broadcast(a, b, 0, NULL, 0, NULL, &broadcasted_a, &broadcasted_b);
    create_tensor(broadcasted_a->shape, broadcasted_a->dimensions, out_tensor);

    for (int i = 0; i < broadcasted_a->total_size; i++) {
        (*out_tensor)->data[i] = broadcasted_a->data[i] + broadcasted_b->data[i];
    }
    
    dispose_tensor(broadcasted_a, true);
    dispose_tensor(broadcasted_b, true);
}
void grad_tensor_add(
    Tensor *a,
    Tensor *b,
    Tensor *grad,
    Tensor **out_grad_a,
    Tensor **out_grad_b
) {
    if (out_grad_a) {
        tensor_reduce(grad, a, out_grad_a);
    }

    if (out_grad_b) {
        tensor_reduce(grad, b, out_grad_b);
    }
}

///////////////////////////////

void tensor_sub(
    Tensor *a,
    Tensor *b,
    Tensor **out_tensor
) {
    Tensor *broadcasted_a = NULL;
    Tensor *broadcasted_b = NULL;
    tensor_broadcast(a, b, 0, NULL, 0, NULL, &broadcasted_a, &broadcasted_b);
    create_tensor(broadcasted_a->shape, broadcasted_a->dimensions, out_tensor);

    for (int i = 0; i < broadcasted_a->total_size; i++) {
        (*out_tensor)->data[i] = broadcasted_a->data[i] - broadcasted_b->data[i];
    }
    
    dispose_tensor(broadcasted_a, true);
    dispose_tensor(broadcasted_b, true);
}
void grad_tensor_sub(
    Tensor *a,
    Tensor *b,
    Tensor *grad,
    Tensor **out_grad_a,
    Tensor **out_grad_b
) {
    if (out_grad_a) {
        tensor_reduce(grad, a, out_grad_a);
    }

    if (out_grad_b) {
        Tensor *neg_grad = NULL;
        tensor_neg(grad, &neg_grad);
        tensor_reduce(neg_grad, b, out_grad_b);
        dispose_tensor(neg_grad, true);
    }
}

///////////////////////////////

void tensor_mul(
    Tensor *a,
    Tensor *b,
    Tensor **out_tensor
) {
    Tensor *broadcasted_a = NULL;
    Tensor *broadcasted_b = NULL;
    tensor_broadcast(a, b, 0, NULL, 0, NULL, &broadcasted_a, &broadcasted_b);
    create_tensor(broadcasted_a->shape, broadcasted_a->dimensions, out_tensor);

    for (int i = 0; i < broadcasted_a->total_size; i++) {
        (*out_tensor)->data[i] = broadcasted_a->data[i] * broadcasted_b->data[i];
    }

    dispose_tensor(broadcasted_a, true);
    dispose_tensor(broadcasted_b, true);
}
void grad_tensor_mul(
    Tensor *a,
    Tensor *b,
    Tensor *grad,
    Tensor **out_grad_a,
    Tensor **out_grad_b
) {
    Tensor *tmp_out_grad_a = NULL, *tmp_out_grad_b = NULL;

    if (out_grad_a) {
        tensor_mul(b, grad, &tmp_out_grad_a);
        tensor_reduce(tmp_out_grad_a, a, out_grad_a);
        dispose_tensor(tmp_out_grad_a, true);
    }

    if (out_grad_b) {
        tensor_mul(a, grad, &tmp_out_grad_b);
        tensor_reduce(tmp_out_grad_b, b, out_grad_b);
        dispose_tensor(tmp_out_grad_b, true);
    }
}

///////////////////////////////

void tensor_div(
    Tensor *a,
    Tensor *b,
    Tensor **out_tensor
) {
    Tensor *broadcasted_a = NULL;
    Tensor *broadcasted_b = NULL;
    tensor_broadcast(a, b, 0, NULL, 0, NULL, &broadcasted_a, &broadcasted_b);
    create_tensor(broadcasted_a->shape, broadcasted_a->dimensions, out_tensor);

    for (int i = 0; i < broadcasted_a->total_size; i++) {
        // Handle the default error case
        if (broadcasted_b->data[i] == 0) {
            fprintf(stderr, "Error: Division by zero error.\n");
            exit(EXIT_FAILURE);
        }
        (*out_tensor)->data[i] = broadcasted_a->data[i] / broadcasted_b->data[i];
    }

    dispose_tensor(broadcasted_a, true);
    dispose_tensor(broadcasted_b, true);
}
void grad_tensor_div(
    Tensor *a,
    Tensor *b,
    Tensor *grad,
    Tensor **out_grad_a,
    Tensor **out_grad_b
) {
    Tensor *tmp_out_grad_a = NULL, *tmp_out_grad_b = NULL;

    if (out_grad_a) {
        Tensor *one = NULL, *one_div_b = NULL;
        create_tensor_from_scalar(1.0, &one);
        tensor_div(one, b, &one_div_b);
        tensor_mul(one_div_b, grad, &tmp_out_grad_a);

        dispose_tensor(one, true);
        dispose_tensor(one_div_b, true);

        tensor_reduce(tmp_out_grad_a, a, out_grad_a);

        dispose_tensor(tmp_out_grad_a, true);
    }

    if (out_grad_b) {
        Tensor *neg_a = NULL, *b_pow_2 = NULL, *neg_a_div_b_pow_2 = NULL;
        tensor_neg(a, &neg_a);
        tensor_mul(b, b, &b_pow_2);
        tensor_div(neg_a, b_pow_2, &neg_a_div_b_pow_2);
        tensor_mul(neg_a_div_b_pow_2, grad, &tmp_out_grad_b);
        
        dispose_tensor(neg_a, true);
        dispose_tensor(b_pow_2, true);
        dispose_tensor(neg_a_div_b_pow_2, true);
        
        tensor_reduce(tmp_out_grad_b, b, out_grad_b);

        dispose_tensor(tmp_out_grad_b, true);
    }

}

///////////////////////////////

void tensor_pow(
    Tensor *a,
    Tensor *b,
    Tensor **out_tensor
) {
    Tensor *broadcasted_a = NULL;
    Tensor *broadcasted_b = NULL;
    tensor_broadcast(a, b, 0, NULL, 0, NULL, &broadcasted_a, &broadcasted_b);
    create_tensor(broadcasted_a->shape, broadcasted_a->dimensions, out_tensor);

    for (int i = 0; i < broadcasted_a->total_size; i++) {
        (*out_tensor)->data[i] = pow(broadcasted_a->data[i], broadcasted_b->data[i]);
    }

    dispose_tensor(broadcasted_a, true);
    dispose_tensor(broadcasted_b, true);
}
void grad_tensor_pow(
    Tensor *a,
    Tensor *b,
    Tensor *out,
    Tensor *grad,
    Tensor **out_grad_a,
    Tensor **out_grad_b
) {
    Tensor *tmp_out_grad_a = NULL, *tmp_out_grad_b = NULL;

    if (out_grad_a) {
        Tensor *one = NULL, *b_sub_one = NULL, *a_pow_b_sub_one = NULL, *b_mul_a_pow_b_sub_one = NULL;
        create_tensor_from_scalar(1.0, &one);
        tensor_sub(b, one, &b_sub_one);
        tensor_pow(a, b_sub_one, &a_pow_b_sub_one);
        tensor_mul(b, a_pow_b_sub_one, &b_mul_a_pow_b_sub_one);
        tensor_mul(b_mul_a_pow_b_sub_one, grad, &tmp_out_grad_a);
        
        dispose_tensor(one, true);
        dispose_tensor(b_sub_one, true);
        dispose_tensor(a_pow_b_sub_one, true);
        dispose_tensor(b_mul_a_pow_b_sub_one, true);

        tensor_reduce(tmp_out_grad_a, a, out_grad_a);
        
        dispose_tensor(tmp_out_grad_a, true);
    }

    if (out_grad_b) {
        Tensor *log_a = NULL, *log_a_mul_out = NULL;
        tensor_log(a, &log_a);
        tensor_mul(log_a, out, &log_a_mul_out);
        tensor_mul(log_a_mul_out, grad, &tmp_out_grad_b);

        dispose_tensor(log_a, true);
        dispose_tensor(log_a_mul_out, true);

        tensor_reduce(tmp_out_grad_b, b, out_grad_b);

        dispose_tensor(tmp_out_grad_b, true);
    }
}

///////////////////////////////

// Function to reshape (or "view") a tensor
void tensor_view(
    Tensor *a,
    Tensor *b,
    bool clone_data,
    Tensor **out_tensor
) {
    // Ensure that the target shape (i.e. "b") is compatible with the original tensor (i.e. "a") size
    if (a->total_size != b->total_size) {
        fprintf(stderr, "Error: New shape is incompatible with the original tensor's data size.\n");
        exit(EXIT_FAILURE);
    }

    // Handle data copying or sharing
    if (clone_data) {
        // Create the new tensor with the specified shape
        create_tensor_like(b, out_tensor);
        // Copy data
        memcpy((*out_tensor)->data, a->data, a->total_size * sizeof(double));
    } else {
        // Create the new tensor with the specified shape
        create_tensor_like_without_data(b, out_tensor);
        // Share the data pointer without allocating new memory
        (*out_tensor)->data = a->data;
    }
}
void grad_tensor_view(
    Tensor *a,
    Tensor *grad,
    Tensor **out_grad_a
) {
    tensor_view(grad, a, true, out_grad_a);
}

///////////////////////////////

typedef struct OpArgs {
    Tensor *a;
    Tensor *b;
    Tensor *out;
    Tensor *grad;
    int dim1;
    int dim2;
    bool clone_data;
} OpArgs;

// This function takes proper parameters and execute the operation
void exec(
    void *op,
    OpArgs op_args,
    Tensor **out_grad_a,
    Tensor **out_grad_b,
    Tensor **out_tensor
) {
         if (op == tensor_sum           ) { tensor_sum(op_args.a, op_args.dim1, false, out_tensor); }
    else if (op == tensor_transpose     ) { tensor_transpose(op_args.a, op_args.dim1, op_args.dim2, op_args.clone_data, out_tensor); }
    else if (op == tensor_matmul        ) { tensor_matmul(op_args.a, op_args.b, out_tensor); }
    else if (op == tensor_softmax       ) { tensor_softmax(op_args.a, op_args.dim1, out_tensor); }
    else if (op == tensor_neg           ) { tensor_neg(op_args.a, out_tensor); }
    else if (op == tensor_log           ) { tensor_log(op_args.a, out_tensor); }
    else if (op == tensor_tan           ) { tensor_tan(op_args.a, out_tensor); }
    else if (op == tensor_tanh          ) { tensor_tanh(op_args.a, out_tensor); }
    else if (op == tensor_exp           ) { tensor_exp(op_args.a, out_tensor); }
    else if (op == tensor_relu          ) { tensor_relu(op_args.a, out_tensor); }
    else if (op == tensor_abs           ) { tensor_abs(op_args.a, out_tensor); }
    else if (op == tensor_add           ) { tensor_add(op_args.a, op_args.b, out_tensor); }
    else if (op == tensor_sub           ) { tensor_sub(op_args.a, op_args.b, out_tensor); }
    else if (op == tensor_mul           ) { tensor_mul(op_args.a, op_args.b, out_tensor); }
    else if (op == tensor_div           ) { tensor_div(op_args.a, op_args.b, out_tensor); }
    else if (op == tensor_pow           ) { tensor_pow(op_args.a, op_args.b, out_tensor); }
    else if (op == tensor_view          ) { tensor_view(op_args.a, op_args.b, op_args.clone_data, out_tensor); }

    else if (op == grad_tensor_sum      ) { grad_tensor_sum(op_args.a, op_args.grad, out_grad_a); }
    else if (op == grad_tensor_transpose) { grad_tensor_transpose(op_args.grad, op_args.dim1, op_args.dim2, out_grad_a); }
    else if (op == grad_tensor_matmul   ) { grad_tensor_matmul(op_args.a, op_args.b, op_args.grad, out_grad_a, out_grad_b); }
    else if (op == grad_tensor_softmax  ) { grad_tensor_softmax(op_args.a, op_args.out, op_args.grad, op_args.dim1, out_grad_a); }
    else if (op == grad_tensor_neg      ) { grad_tensor_neg(op_args.grad, out_grad_a); }
    else if (op == grad_tensor_log      ) { grad_tensor_log(op_args.a, op_args.grad, out_grad_a); }
    else if (op == grad_tensor_tan      ) { grad_tensor_tan(op_args.a, op_args.grad, out_grad_a); }
    else if (op == grad_tensor_tanh     ) { grad_tensor_tanh(op_args.out, op_args.grad, out_grad_a); }
    else if (op == grad_tensor_exp      ) { grad_tensor_exp(op_args.out, op_args.grad, out_grad_a); }
    else if (op == grad_tensor_relu     ) { grad_tensor_relu(op_args.a, op_args.grad, out_grad_a); }
    else if (op == grad_tensor_abs      ) { grad_tensor_abs(op_args.a, op_args.grad, out_grad_a); }
    else if (op == grad_tensor_add      ) { grad_tensor_add(op_args.a, op_args.b, op_args.grad, out_grad_a, out_grad_b); }
    else if (op == grad_tensor_sub      ) { grad_tensor_sub(op_args.a, op_args.b, op_args.grad, out_grad_a, out_grad_b); }
    else if (op == grad_tensor_mul      ) { grad_tensor_mul(op_args.a, op_args.b, op_args.grad, out_grad_a, out_grad_b); }
    else if (op == grad_tensor_div      ) { grad_tensor_div(op_args.a, op_args.b, op_args.grad, out_grad_a, out_grad_b); }
    else if (op == grad_tensor_pow      ) { grad_tensor_pow(op_args.a, op_args.b, op_args.out, op_args.grad, out_grad_a, out_grad_b); }
    else if (op == grad_tensor_view     ) { grad_tensor_view(op_args.a, op_args.grad, out_grad_a); }
    
    else if (op == (void (*) (OpArgs, Tensor **, Tensor **, Tensor **)) op) { 
        void (*custom_op) (OpArgs, Tensor **, Tensor **, Tensor **) = (void (*) (OpArgs, Tensor **, Tensor **, Tensor **)) op;
        custom_op(op_args, out_grad_a, out_grad_b, out_tensor);
    } else {
        fprintf(stderr, "Error: Operation is not supported.\n");
        exit(EXIT_FAILURE);
    }
}

/////////////////////////////////////////////////////////////////////

// Define a structure for a node in the queue
typedef struct QNode {
    void *data;
    struct QNode *next;
} QNode;

// Define the queue structure
typedef struct Queue {
    QNode *front; // Pointer to the front node
    QNode *rear;  // Pointer to the rear node
    int size;     // Number of elements in the queue
} Queue;

// Function to create a new queue
Queue* create_queue() {
    Queue *queue = (Queue*)malloc(sizeof(Queue));
    if (!queue) {
        fprintf(stderr, "Failed to create queue");
        exit(EXIT_FAILURE);
    }
    queue->front = NULL;
    queue->rear = NULL;
    queue->size = 0;
    return queue;
}

// Function to push an element into the queue
void push_queue(
    Queue *queue,
    void *data
) {
    if (!queue) {
        perror("Queue not initialized");
        return;
    }

    QNode *new_node = (QNode*)malloc(sizeof(QNode));
    if (!new_node) {
        fprintf(stderr, "Failed to allocate memory for new node");
        exit(EXIT_FAILURE);
    }

    new_node->data = data;
    new_node->next = NULL;

    if (queue->rear) {
        queue->rear->next = new_node;
    }
    queue->rear = new_node;

    // If the queue was empty, the new node is also the front
    if (!queue->front) {
        queue->front = new_node;
    }

    queue->size++;
}

// Function to pop an element from the queue
void* pop_queue(
    Queue *queue
) {
    if (!queue || !queue->front) {
        return NULL;
    }

    QNode *temp = queue->front;
    void *data = temp->data;

    queue->front = temp->next;

    // If the queue is now empty, update the rear to NULL
    if (!queue->front) {
        queue->rear = NULL;
    }

    free(temp);
    queue->size--;

    return data;
}

// Function to free the queue
void dispose_queue(
    Queue *queue
) {
    while (queue->size != 0) {
        pop_queue(queue);
    }
    free(queue);
}

/////////////////////////////////////////////////////////////////////

// Computational graph node structure
typedef struct Node {
    bool is_leaf;          // Is leaf node
    bool is_param;         // Is parameter
    OpArgs args;           // Operation arguments
    void *forward_op;      // Forward function
    void *backward_op;     // Backward function
    Tensor *value;         // Forward value
    Tensor *grad;          // Gradient of the node
    struct Node *input_a;  // Input node a
    struct Node *input_b;  // Input node b
    int ref_count;         // Reference count for backward pass
    bool requires_grad;    // Whether this node requires gradient
} Node;

void dispose_node(
    Node *node
) {
    if (node->value) { dispose_tensor(node->value, true); node->value = NULL; }
    if (node->grad)  { dispose_tensor(node->grad, true);  node->grad  = NULL; }
    free(node);
}

void __create_node(
    bool is_leaf,
    bool is_param,
    void *forward_op,
    void *backward_op,
    int dim1,
    int dim2,
    Tensor *value,
    Node *input_a,
    Node *input_b,
    bool requires_grad,
    Node **out_node
) {

    Node *node = (Node *)malloc(sizeof(Node));

    if (is_leaf) {
        node->is_leaf = is_leaf;
        node->is_param = is_param;
        node->value = value;
        node->requires_grad = requires_grad;

        node->forward_op = NULL;
        node->backward_op = NULL;
        node->grad = NULL;
        node->input_a = NULL;
        node->input_b = NULL;
        node->ref_count = 0;
    } else {
        node->is_leaf = is_leaf;
        node->is_param = is_param;
        node->forward_op = forward_op;
        node->backward_op = backward_op;
        node->grad = NULL;
        node->input_a = input_a;
        node->input_b = input_b;
        node->ref_count = 0;
        node->requires_grad = (input_a && input_a->requires_grad) || (input_b && input_b->requires_grad);

        node->args.dim1 = dim1;
        node->args.dim2 = dim2;
        node->args.clone_data = true; // Always clone data for better memory-management while disposing data
        if (input_a) { node->args.a = node->input_a->value; node->input_a->ref_count++; }
        if (input_b) { node->args.b = node->input_b->value; node->input_b->ref_count++; }

        exec(node->forward_op, node->args, NULL, NULL, &(node->value));
    }

    if (out_node) *out_node = node;
}

void create_leaf(
    Tensor *value,
    bool requires_grad,
    Node **out_node
) {
    __create_node(true, false, NULL, NULL, -1, -1, value, NULL, NULL, requires_grad, out_node);
}

void create_param(
    Queue *param_list,
    Tensor *value,
    Node **out_node
) {
    __create_node(true, true, NULL, NULL, -1, -1, value, NULL, NULL, true, out_node);
    push_queue(param_list, *out_node);
}

void create_n_exec_op(
    void *forward_op,
    void *backward_op,
    Node *input_a,
    Node *input_b,
    Node **out_node
) {
    __create_node(false, false, forward_op, backward_op, -1, -1, NULL, input_a, input_b, /* Don't care!*/ false, out_node);
}

void create_n_exec_op_1_dim(
    void *forward_op,
    void *backward_op,
    int dim,
    Node *input_a,
    Node *input_b,
    Node **out_node
) {
    __create_node(false, false, forward_op, backward_op, dim, -1, NULL, input_a, input_b, /* Don't care!*/ false, out_node);
}

void create_n_exec_op_2_dim(
    void *forward_op,
    void *backward_op,
    int dim1,
    int dim2,
    Node *input_a,
    Node *input_b,
    Node **out_node
) {
    __create_node(false, false, forward_op, backward_op, dim1, dim2, NULL, input_a, input_b, /* Don't care!*/ false, out_node);
}

void backward(Node *loss_node) {
    Queue *queue = create_queue();
    push_queue(queue, loss_node);

    Tensor *loss_grad = NULL;
    create_tensor_like(loss_node->value, &loss_grad);
    init_tensor(1.0, loss_grad);
    loss_node->grad = loss_grad;

    while (queue->size > 0) {
        Node *node = (Node *)pop_queue(queue);

        if (node->backward_op && node->requires_grad) {
            Node *inp_a = node->input_a;
            Node *inp_b = node->input_b;

            Tensor *grad_a = NULL, *grad_b = NULL;
            node->args.out = node->value;
            node->args.grad = node->grad;
            exec(node->backward_op, node->args, &grad_a, &grad_b, NULL);

            if (inp_a && inp_a->requires_grad && inp_a->grad && grad_a) {
                Tensor *tmp_grad_a = NULL;
                tensor_add(inp_a->grad, grad_a, &tmp_grad_a);
                memcpy(inp_a->grad->data, tmp_grad_a->data, tmp_grad_a->total_size * sizeof(double));
                dispose_tensor(grad_a, true);
                dispose_tensor(tmp_grad_a, true);
            } else if (inp_a && inp_a->requires_grad && grad_a) {
                inp_a->grad = grad_a;
            }

            if (inp_b && inp_b->requires_grad && inp_b->grad && grad_b) {
                Tensor *tmp_grad_b = NULL;
                tensor_add(inp_b->grad, grad_b, &tmp_grad_b);
                memcpy(inp_b->grad->data, tmp_grad_b->data, tmp_grad_b->total_size * sizeof(double));
                dispose_tensor(grad_b, true);
                dispose_tensor(tmp_grad_b, true);
            } else if (inp_b && inp_b->requires_grad && grad_b) {
                inp_b->grad = grad_b;
            }

            inp_a->ref_count--;
            if (inp_a->ref_count == 0) {
                push_queue(queue, inp_a);
            }
            
            if (inp_b) {
                inp_b->ref_count--;
                if (inp_b->ref_count == 0) {
                    push_queue(queue, inp_b);
                }
            }
        }
    }

    dispose_queue(queue);
}

void update_params(Queue *param_list, Tensor *lr) {
    QNode *current = param_list->front;

    while(current != NULL) {
        Node *current_data = (Node *)current->data;
        Tensor *delta = NULL, *new_value = NULL;
        tensor_mul(current_data->grad, lr, &delta);
        tensor_sub(current_data->value, delta, &new_value);
        memcpy(current_data->value->data, new_value->data, new_value->total_size * sizeof(double));
        dispose_tensor(delta, true);
        dispose_tensor(new_value, true);
        current = current->next;
    }
}

void zero_grad(Queue *param_list) {
    QNode *current = param_list->front;

    while(current != NULL) {
        Node *current_data = (Node *)current->data;
        if (current_data->grad) {
            dispose_tensor(current_data->grad, true);
            current_data->grad = NULL;
        }
        current = current->next;
    }
}

void dispose_graph(Node *loss_node) {
    Queue *queue = create_queue();
    push_queue(queue, loss_node);

    while (queue->size > 0) {
        Node *node = (Node *)pop_queue(queue);
        Node *inp_a = node->input_a;
        Node *inp_b = node->input_b;

        if (inp_a && inp_a->forward_op) push_queue(queue, inp_a);
        if (inp_b && inp_b->forward_op) push_queue(queue, inp_b);
        dispose_node(node);
    }

    dispose_queue(queue);
}

/////////////////////////////////////////////////////////////////////

typedef struct LinearLayer {
    bool bias;
    int input_feature_size;
    int output_feature_size;

    Node *n_W;
    Node *n_b;
} LinearLayer;

LinearLayer *linearlayer(
    Queue *param_list,
    int input_feature_size,
    int output_feature_size,
    bool bias
) {
    LinearLayer *ll = (LinearLayer *)malloc(sizeof(LinearLayer));
    
    ll->bias = bias;
    ll->input_feature_size = input_feature_size;
    ll->output_feature_size = output_feature_size;

    Tensor *W = NULL;
    create_tensor((int[]) {ll->output_feature_size, ll->input_feature_size}, 2, &W);
    init_tensor_rand(1.0 / ll->input_feature_size, W);
    create_param(param_list, W, &(ll->n_W));

    if (bias) {
        Tensor *b = NULL;
        create_tensor((int[]) {1, ll->output_feature_size}, 2, &b);
        init_tensor_rand(1.0 / ll->input_feature_size, b);
        create_param(param_list, b, &(ll->n_b));
    }

    return ll;
}

void forward_linearlayer(
    LinearLayer *ll,
    Node *n_X,
    Node **out_node
) {
    Node *n_transpose = NULL, *n_matmul = NULL;
    create_n_exec_op_2_dim(tensor_transpose, grad_tensor_transpose, 0, 1, ll->n_W, NULL, &n_transpose);
    create_n_exec_op(tensor_matmul, grad_tensor_matmul, n_X, n_transpose, &n_matmul);
    *out_node = n_matmul;

    if (ll->bias) {
        Node *n_X_matmul_W_T_add_b = NULL;
        create_n_exec_op(tensor_add, grad_tensor_add, n_matmul, ll->n_b, out_node);
    }
}

void dispose_linearlayer(LinearLayer *ll) {
    dispose_node(ll->n_W); ll->n_W = NULL;
    if (ll->bias) dispose_node(ll->n_b); ll->n_b = NULL;
    free(ll);
}

/////////////////////////////////////////////////////////////////////

typedef struct ActivationLayer {
    void *forward_op;
    void *backward_op;
} ActivationLayer;

ActivationLayer *activation_layer(
    void *forward_op,
    void *backward_op
) {
    ActivationLayer *al = (ActivationLayer *)malloc(sizeof(ActivationLayer));
    al->forward_op = forward_op;
    al->backward_op = backward_op;

    return al;
}

void forward_activationlayer(
    ActivationLayer *al,
    Node *n_X,
    Node **out_node
) {
    create_n_exec_op(al->forward_op, al->backward_op, n_X, NULL, out_node);
}

void forward_activationlayer_with_dim(
    ActivationLayer *al,
    Node *n_X,
    int dim,
    Node **out_node
) {
    create_n_exec_op_1_dim(al->forward_op, al->backward_op, dim, n_X, NULL, out_node);
}

void dispose_activationlayer(ActivationLayer *activationlayer) {
    free(activationlayer);
}

/////////////////////////////////////////////////////////////////////

typedef struct MSELoss {
    Node *n_c;
    Node *n_two;
} MSELoss;

MSELoss *mseloss() {
    MSELoss *mseloss = (MSELoss *)malloc(sizeof(MSELoss));
    
    Tensor *c = NULL, *two = NULL;
    create_tensor_from_scalar(1.0, &c);
    create_tensor_from_scalar(2.0, &two);
    create_leaf(c, false, &(mseloss->n_c));
    create_leaf(two, false, &(mseloss->n_two));

    return mseloss;
}

void forward_mseloss(
    MSELoss *mseloss,
    Node *n_X,
    Node *n_target,
    Node **out_node
) {
    mseloss->n_c->value->data[0] = 1.0 / (n_X->value->shape[n_X->value->dimensions - 1] * n_X->value->shape[n_X->value->dimensions - 2]);

    Node *n_sub = NULL, *n_pow = NULL, *n_mul = NULL;
    create_n_exec_op(tensor_sub, grad_tensor_sub, n_X, n_target, &n_sub);
    create_n_exec_op(tensor_pow, grad_tensor_pow, n_sub, mseloss->n_two, &n_pow);
    create_n_exec_op(tensor_mul, grad_tensor_mul, n_pow, mseloss->n_c, &n_mul);
    create_n_exec_op_1_dim(tensor_sum, grad_tensor_sum, -1, n_mul, NULL, out_node);
}

void dispose_mseloss(MSELoss *mseloss) {
    dispose_node(mseloss->n_c); mseloss->n_c = NULL;
    dispose_node(mseloss->n_two); mseloss->n_two = NULL;
    free(mseloss);
}

typedef struct CrossEntropyLoss {
    Node *n_c;
} CrossEntropyLoss;

CrossEntropyLoss *crossentropyloss() {
    CrossEntropyLoss *crossentropyloss = (CrossEntropyLoss *)malloc(sizeof(CrossEntropyLoss));
    
    Tensor *c = NULL;
    create_tensor_from_scalar(1.0, &c);
    create_leaf(c, false, &(crossentropyloss->n_c));

    return crossentropyloss;
}

void grad_tensor_softmax_for_crossentropyloss(
    OpArgs op_args,
    Tensor **out_grad_a,
    Tensor **out_grad_b,
    Tensor **out_tensor
) {
    Tensor *tmp_out_grad_a = NULL;

    tensor_sub(op_args.b, op_args.out, &tmp_out_grad_a);
    tensor_mul(tmp_out_grad_a, op_args.grad, out_grad_a);

    dispose_tensor(tmp_out_grad_a, true);
}

void tensor_crossentropyloss(
    OpArgs op_args,
    Tensor **out_grad_a,
    Tensor **out_grad_b,
    Tensor **out_tensor
) {
    Tensor *log = NULL;
    
    tensor_log(op_args.a, &log);
    tensor_mul(log, op_args.b, out_tensor);

    dispose_tensor(log, true);
}
void grad_tensor_crossentropyloss(
    OpArgs op_args,
    Tensor **out_grad_a,
    Tensor **out_grad_b,
    Tensor **out_tensor
) {
    deep_copy_tensor(op_args.grad, out_grad_a);
}

void forward_crossentropyloss(
    CrossEntropyLoss *crossentropyloss,
    Node *n_X,
    Node *n_target,
    int dim,
    Node **out_node
) {
    // Always dim[0] is the "batch size" no matter what, look at PyTorch documentation.
    int batch_size = n_X->value->shape[0];
    crossentropyloss->n_c->value->data[0] = 1.0 / batch_size;

    Node *n_softmax = NULL, *n_cel = NULL, *n_sum = NULL, *n_mul = NULL;
    create_n_exec_op_1_dim(tensor_softmax, grad_tensor_softmax_for_crossentropyloss, dim, n_X, n_target, &n_softmax);
    create_n_exec_op(tensor_crossentropyloss, grad_tensor_crossentropyloss, n_softmax, n_target, &n_cel);
    create_n_exec_op_1_dim(tensor_sum, grad_tensor_sum, -1, n_cel, NULL, &n_sum);
    create_n_exec_op(tensor_mul, grad_tensor_mul, n_sum, crossentropyloss->n_c, &n_mul);
    create_n_exec_op(tensor_neg, grad_tensor_neg, n_mul, NULL, out_node);
}

void dispose_crossentropyloss(CrossEntropyLoss *crossentropyloss) {
    dispose_node(crossentropyloss->n_c); crossentropyloss->n_c = NULL;
    free(crossentropyloss);
}

/////////////////////////////////////////////////////////////////////

// Function to shuffle data in a batch of dataset
void shuffle_data(
    Tensor *X,
    Tensor *Y
) {
    int batch_size = X->shape[0];
    int x_feature_size = X->shape[1];
    int y_feature_size = Y->shape[1];

    // Fisher-Yates shuffle algorithm
    for (int i = batch_size - 1; i > 0; i--) {
        // Generate a random index j such that 0 <= j <= i
        int j = rand() % (i + 1);

        // Swap row i with row j in X
        for (int k = 0; k < x_feature_size; k++) {
            double temp = get_element(X, i, k);
            set_element(X, get_element(X, j, k), i, k);
            set_element(X, temp, j, k);
        }

        // Swap the corresponding row i with row j in Y
        for (int k = 0; k < y_feature_size; k++) {
            double temp = get_element(Y, i, k);
            set_element(Y, get_element(Y, j, k), i, k);
            set_element(Y, temp, j, k);
        }
    }
}

// Function to pick an index with modified likelihood
int pick_random_index(
    double probabilities[],
    int n,
    double power
) {
    // Step 1: Apply the power to increase the likelihood of higher probabilities
    double modified_probs[n];
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        modified_probs[i] = pow(probabilities[i], power);
        sum += modified_probs[i];
    }

    // Step 2: Normalize the modified probabilities
    for (int i = 0; i < n; i++) {
        modified_probs[i] /= sum;
    }

    // Step 3: Create a cumulative probability array
    double cumulative_prob[n];
    cumulative_prob[0] = modified_probs[0];
    for (int i = 1; i < n; i++) {
        cumulative_prob[i] = cumulative_prob[i - 1] + modified_probs[i];
    }

    // Step 4: Generate a random number between 0 and 1
    double random_value = (double)rand() / RAND_MAX;

    // Step 5: Find the first index where cumulative probability is >= random_value
    for (int i = 0; i < n; i++) {
        if (random_value <= cumulative_prob[i]) {
            return i;
        }
    }

    // Fallback (shouldn't happen if probabilities sum to 1)
    return n - 1;
}

/////////////////////////////////////////////////////////////////////

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

// /////////////////////////////////////////////////////////////////////

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

    CrossEntropyLoss *cel = crossentropyloss();

    Node *n_x = NULL, *n_y = NULL, *n_loss = NULL;
    create_leaf(x, false, &n_x); n_x->requires_grad = true;
    create_leaf(y, false, &n_y);
    forward_crossentropyloss(cel, n_x, n_y, 1, &n_loss);

    print_info(n_x->value);
    print_info(n_y->value);
    print_info(n_loss->value);

    backward(n_loss);

    print_info(n_x->grad);
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

    // mnist_train("./mnist_train_small.csv");
    // mnist_eval("./mnist_test.csv");

    // bigram_train("./names.txt");
    // bigram_test();
    
    // mlp_train("./names.txt");
    // mlp_test();

    // simple_test();

    // simple_test2();

    return 0;
}