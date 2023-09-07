#include "ModelParser.h"

const value_t *ModelParser_GetAt(ModelParser_t *self, size_t size) {
    const value_t *at = self->at;
    self->at += size % 32 == 0 ? size: size + (32 - size % 32);
    assert(self->at <= self->end);
    return at;
}
