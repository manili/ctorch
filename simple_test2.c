#include "ctorch.c"

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

int main(
    int argc,
    char *argv[]
) {
    setup_application(42);

    simple_test2();

    return 0;
}