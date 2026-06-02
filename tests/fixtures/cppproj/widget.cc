// Token for E2E search: CPLUSPLUS_FIXTURE_E2E_TOKEN_X1
#include "widget.h"

namespace widgets {

Widget::Widget(int size) : size_(size) {
    // Initialise with the requested size; clamp to a sane range.
    if (size_ < 0) {
        size_ = 0;
    }
}

int Widget::area() const {
    // Compute the squared area of this widget.
    return size_ * size_;
}

void Widget::resize(int new_size) {
    // Resize the widget; this preserves the existing identity.
    size_ = new_size > 0 ? new_size : 0;
}

}  // namespace widgets
