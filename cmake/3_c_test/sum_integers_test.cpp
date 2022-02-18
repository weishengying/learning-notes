#include "sum_integers.hpp"
#include <gtest/gtest.h>

using namespace std; 

TEST(IsAbsTest, HandlerTrueReturn)
{
  auto integers = {1, 2, 3, 4, 5};
  ASSERT_TRUE(sum_integers(integers) == 15)  << "sum_integers(integers) == 15";//ASSERT_TRUE期待结果是true,operator<<输出一些自定义的信息
}

int main(int argc, char* argv[])
{
	testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}