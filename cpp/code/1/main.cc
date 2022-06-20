#include "op_registry.h"

int main() { 
  std::cout << opregistry::instance().creatOp("custom_op_1") << std::endl;
  std::cout << opregistry::instance().creatOp("new_custom_op_1") << std::endl;
  
}