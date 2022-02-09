#include<iostream>
#include<gtest/gtest.h>

using namespace std; 

int Abs(int x)
{
     return x > 0 ? x : -x;
}

TEST(IsAbsTest, HandlerTrueReturn)
{
    ASSERT_TRUE(Abs(1) == 1) << "Abs(1)=1";  //ASSERT_TRUE期待结果是true,operator<<输出一些自定义的信息
    ASSERT_TRUE(Abs(-1) == 1) << "Abs(-1)=1";
    ASSERT_FALSE(Abs(-2) == -2);  //期待结果是false
    ASSERT_EQ(Abs(1),Abs(-1));
    ASSERT_NE(Abs(-1),0);
    ASSERT_LT(Abs(-1),2);
    ASSERT_GT(Abs(-1),0);
    ASSERT_LE(Abs(-1),2);
    ASSERT_GE(Abs(-1),0);
}

int main(int argc, char* argv[])
{
	testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}