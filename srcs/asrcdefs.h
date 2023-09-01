#pragma once

#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef enum { FALSE = 0, TRUE = 1 } bool_t;
typedef enum { S_BEGIN, S_MIDDLE, S_END, S_ALL, S_ERR } flag_t;
typedef float value_t;
