#include "log.h"
#include <stdarg.h>
#include <openssl/evp.h>

void dolog(const char *format, ...) {
    va_list ap;
    printf("LOG: ");
    va_start(ap, format);
    vprintf(format, ap);
    va_end(ap);
    fflush(stdout);
}

void logdigest(const char *prefix, const void *body, size_t body_size) {
    char hexdigest[EVP_MAX_MD_SIZE * 2 + 1];
    EVP_MD_CTX *ctx; unsigned char digest[EVP_MAX_MD_SIZE]; unsigned digest_size;
    ctx = EVP_MD_CTX_new();
    EVP_DigestInit_ex(ctx, EVP_md5(), NULL);
    EVP_DigestUpdate(ctx, body, body_size);
    EVP_DigestFinal_ex(ctx, digest, &digest_size);
    EVP_MD_CTX_free(ctx);
    for (unsigned index = 0; index < digest_size; ++index)
	sprintf(hexdigest + index + index, "%02X", (unsigned)digest[index]);
    dolog("{ \"name\": \"%s\", \"digest\": \"%s\" }\n", prefix, hexdigest);
}

void bin_dump(const char *prefix, size_t *count, const void *body, size_t body_size) {
    char fpath[128];
    snprintf(fpath, sizeof(fpath), "dumps/%s.%04zu.bin", prefix, (*count)++);
    FILE *wfp = fopen(fpath, "wb");
    fwrite(body, 1, body_size, wfp);
    fclose(wfp);
}
void bin_load(const char *prefix, size_t *count, void *body, size_t body_size) {
    char fpath[128];
    snprintf(fpath, sizeof(fpath), "dumps/%s.%04zu.bin", prefix, (*count)++);
    FILE *rfp = fopen(fpath, "rb");
    fread(body, 1, body_size, rfp);
    fclose(rfp);
}
