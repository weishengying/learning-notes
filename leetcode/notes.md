# 1 回溯算法

[二叉树的深度](https://leetcode.cn/problems/er-cha-shu-de-shen-du-lcof/submissions/)

```cpp
class Solution {
public:
    void dfs(TreeNode* root, int& depth, int& max) {
        if(!root)
            return;
        depth += 1;
        if(depth > max)
            max = depth;
        if(root->left){
            dfs(root->left, depth, max);
            depth -= 1;
        }
        if(root->right){
            dfs(root->right, depth, max);
            depth -= 1;
        }
    }
    int maxDepth(TreeNode* root) {
        int depth = 0;
        int max = 0;
        dfs(root, depth, max);
        return max;
    }
};
```

或者

```cpp
class Solution {
public:
    void dfs(TreeNode* root, int depth, int& max) {
        if(!root)
            return;
        depth += 1;
        if(depth > max)
            max = depth;
        if(root->left){
            dfs(root->left, depth, max);
        }
        if(root->right){
            dfs(root->right, depth, max);
        }
    }
    int maxDepth(TreeNode* root) {
        int depth = 0;
        int max = 0;
        dfs(root, depth, max);
        return max;
    }
};
```
