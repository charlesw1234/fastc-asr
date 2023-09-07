#include "alignedmem.hxx"
#include "log.h"


void *aligned_malloc(size_t alignment, size_t required_bytes)
{
    void *p1;  // original block
    void **p2; // aligned block
    int offset = alignment - 1 + sizeof(void *);
    //printf("aligned_malloc.%u: required_bytes = %zu, alignment = %zu\n",
    //__LINE__, required_bytes, alignment);
    if ((p1 = (void *)malloc(required_bytes + offset)) == NULL) {
        return NULL;
    }
    p2 = (void **)(((size_t)(p1) + offset) & ~(alignment - 1));
    p2[-1] = p1;
    //printf("aligned_malloc.%u: p1 = %p, p2 = %p\n", __LINE__, p1, p2);
    return p2;
}

void aligned_free(void *p)
{
    free(((void **)p)[-1]);
}
