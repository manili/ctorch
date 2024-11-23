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