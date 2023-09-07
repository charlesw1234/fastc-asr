#include "TensorValue.h"
#include <openssl/evp.h>

static inline void alloc_body(TensorValue_t *self, size_t num_values) {
    self->body = (value_t *)malloc(sizeof(value_t) * num_values);
}

void TensorValue_Init(TensorValue_t *self, size_t a, size_t b, size_t c, size_t d) {
    self->size[0] = a; self->size[1] = b; self->size[2] = c; self->size[3] = d;
    alloc_body(self, TensorValue_NumValues(self));
}
void TensorValue_CopyInit(TensorValue_t *self, const TensorValue_t *in) {
    memcpy(self->size, in->size, sizeof(self->size));
    size_t num_values = TensorValue_NumValues(self);
    alloc_body(self, num_values);
    memcpy(self->body, in->body, sizeof(value_t) * num_values);
}
void TensorValue_Zero(TensorValue_t *self) {
    memset(self->body, 0, sizeof(value_t) * TensorValue_NumValues(self));
}
void TensorValue_Add(TensorValue_t *self, value_t coe, const TensorValue_t *in) {
    size_t at, num_values = TensorValue_NumValues(self);
    assert(memcmp(self->size, in->size, sizeof(self->size)) == 0);
    for (at = 0; at < num_values; ++at) self->body[at] += coe * in->body[at];
}

void TensorValue_DoLog(const TensorValue_t *self, const char *prefix) {
    char sizestr[32], hexdigest[EVP_MAX_MD_SIZE * 2 + 1];
    EVP_MD_CTX *ctx; unsigned char digest[EVP_MAX_MD_SIZE]; unsigned digest_size;
    snprintf(sizestr, sizeof(sizestr), "[%zu, %zu, %zu, %zu]",
	     self->size[0], self->size[1], self->size[2], self->size[3]);
    ctx = EVP_MD_CTX_new();
    EVP_DigestInit_ex(ctx, EVP_md5(), NULL);
    EVP_DigestUpdate(ctx, self->body, sizeof(value_t) * TensorValue_NumValues(self));
    EVP_DigestFinal_ex(ctx, digest, &digest_size);
    EVP_MD_CTX_free(ctx);
    for (size_t index = 0; index < digest_size; ++index)
	sprintf(hexdigest + index + index, "%02X", (unsigned)digest[index]);
    dolog("[\"%s\", %s, \"%s\"]\n", prefix, sizestr, hexdigest);
}
