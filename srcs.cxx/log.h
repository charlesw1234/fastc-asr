#pragma once

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif
    void dolog(const char *format, ...);
    void logdigest(const char *prefix, const void *body, size_t body_size);

    void bin_dump(const char *prefix, size_t *count, const void *body, size_t body_size);
    void bin_load(const char *prefix, size_t *count, void *body, size_t body_size);
#ifdef __cplusplus
}
#endif
