
#ifndef MODEL_H
#define MODEL_H

#include <string>
#include <stdio.h>

class Model {
  public:
    virtual ~Model(){};
    virtual void reset() = 0;
    virtual std::string forward_chunk(float *din, int len, int flag) = 0;
    virtual std::string forward(float *din, int len, int flag) = 0;
    virtual std::string rescoring() = 0;
};

Model *create_model(const char *path, int mode);
#endif
