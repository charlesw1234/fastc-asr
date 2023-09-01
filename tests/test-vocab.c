#include "Vocab.h"
#include <stdio.h>

int main(void) {
    Vocab_t vocab;
    printf("Init: %s\n", Vocab_Init(&vocab, "models/k2_rnnt2_cli/vocab.txt")
	   ? "True": "False");
    printf("Vocab size: %u\n", (unsigned)Vocab_Size(&vocab));
    char strbuf[256];
    int vector[] = { 160, 373, 873, 2551, 238, 437, 918 };
    printf("Vector2String: %s\n", (int)
	   Vocab_Vector2String(&vocab, strbuf, sizeof(strbuf),
			       vector, sizeof(vector) / sizeof(vector[0]))
	   ? "True": "False");
    printf("strbuf: %s\n", strbuf);
    Vocab_Destroy(&vocab);
    return 0;
}
