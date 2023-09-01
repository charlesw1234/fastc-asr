#include "Vocab.h"
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>

bool_t Vocab_Init(Vocab_t *self, const char *vocab_fpath) {
    struct stat st;
    if ((self->fd = open(vocab_fpath, O_RDONLY)) < 0) goto errquit0;
    fstat(self->fd, &st); self->fsize = st.st_size;
    self->start = (char *)mmap(NULL, self->fsize, PROT_READ, MAP_SHARED, self->fd, 0);
    if (self->start == NULL) goto errquit1;
    const char *last = self->start + self->fsize;
    self->nvocabs = 0;
    for (const char *cur = self->start; cur < last; ++cur)
	if (*cur == '\n') ++self->nvocabs;
    self->vocab_starts = (const char **)malloc(sizeof(char *) * self->nvocabs);
    if (self->vocab_starts == NULL) goto errquit2;
    size_t at = 0; self->vocab_starts[at++] = self->start;
    for (const char *cur = self->start; cur < last; ++cur)
	if (*cur == '\n' && cur + 1 < last && at + 1 < self->nvocabs)
	    self->vocab_starts[at++] = cur + 1;
    return TRUE;
 errquit2: munmap(self->start, self->fsize);
 errquit1: close(self->fd);
 errquit0: return FALSE;
}
void Vocab_Destroy(Vocab_t *self) {
    free(self->vocab_starts);
    munmap(self->start, self->fsize);
    close(self->fd);
}

bool_t Vocab_Vector2String(const Vocab_t *self, char *strbuf, size_t strbuf_size,
			   const int *hyps, size_t hyps_size) {
    const char *vocab_cur;
    char *cur = strbuf, *last = strbuf + strbuf_size;
    const int *hyp_cur, *hyp_last = hyps + hyps_size;
    for (hyp_cur = hyps; hyp_cur < hyp_last; ++hyp_cur) {
	assert(*hyp_cur >= 0 && *hyp_cur < self->nvocabs);
	vocab_cur = self->vocab_starts[*hyp_cur];
	while (*vocab_cur != '\n' && cur < last) *cur++ = *vocab_cur++;
    }
    if (cur == last) return FALSE;
    *cur++ = 0;
    return TRUE;
}
