# KDTree
**KDTree的定义为:**
```cpp
template <typename T>
class KDTree
{
public:
    KDTree() { root_node = nullptr; }
    KDTree(std::vector<T *> _data_ptr_list, int _split_dims = -1);
    KDTree(std::vector<T> _data_list, int _split_dims = -1);
    ~KDTree();

    void setData(std::vector<T *> _data_ptr_list, int _split_dims = -1);
    void insertNode(T *_node);
    void removeNode(T *_node);

    T *getNearestPoint(T *_pt);
    std::vector<T *> getPointsInRange(T *_pt, float _range, int _k = -1);
    std::vector<T *> getKNearestPoints(T *_pt, int _k);

protected:
    void getPointsInRangeInBranch(T *_pt, float _range, Node<T> *_branch_node, Node<T> *_start_node = nullptr, int _k = -1);
    void getKNearestPointsInBranch(T *_pt, int _k, Node<T> *_branch_node, Node<T> *_start_node = nullptr);

    Node<T> *findNearestLeaf(T *_pt, Node<T> *_branch_node);
    Node<T> *findAnotherSuBranch(Node<T> *_parent, Node<T> *_sub_node);

    void clear();
    Node<T> *buildTree(std::vector<T *> *_data_list, Node<T> *_parent_node);
    void deleteBranch(Node<T> *_root_node);

    Eigen::VectorXf calcVarience(std::vector<T *> *_datas);
    bool compareByDim(const T *_a, const T *_b, int _dim);
    void sort_vector_list(std::vector<T *> *_datas, int splid_dim);

public:
    std::vector<T> data_list;
    std::vector<T *> data_ptr_list;
    int split_dim;
    Node<T> *root_node;
    int deactivate_nodes_num;

private:
    std::vector<float> k_nearest_dists;
    std::vector<T *> k_nearest_pts;
};
```
其中T为任意 <span style="color:red">包含数据成员为 Eigen::VectorXf data</span> 的类, 划分维数时只考虑[0, _split_dims], 默认为全部数据.



**KDTree的Node定义为:**
```cpp
template <typename T>
class Node
{
public:
    Node(T *_param, int _split_dim) : param(_param), split_dim(_split_dim)
    {
        left = nullptr;
        right = nullptr;
        parent = nullptr;

        deactivate = false;
    }
    ~Node() { ; }

    T *param;
    int split_dim;
    bool deactivate;

    Node<T> *left;
    Node<T> *right;
    Node<T> *parent;
};
```

## 使用例程
```cpp
#include <eigen3/Eigen/Dense>
#include <iostream>

#include "KDTree/KDTree.h"
#include "KDTree/KDTree.cpp"

template <typename T>
void display(std::vector<T> input)
{
  for (auto it : input)
  {
    std::cout << it << " ";
  }
  std::cout << std::endl;
}

class param
{
public:
    param(Eigen::VectorXf _data) : data(_data) { ; }
    ~param() { ; }
    Eigen::VectorXf data; // 一定要包含
};

int main()
{
    int length = 10000;
    int dim = 100;

    std::vector<param> test_datas;
    std::vector<float> len_list;

    Eigen::VectorXf temp(dim);
    Eigen::VectorXf test(dim);
    test.setRandom() * 5;

    for (int i = 0; i < length; i++)
    {
        temp.setRandom() * 10;
        test_datas.push_back(param(temp));
        len_list.push_back(std::sqrt((test[0] - temp[0]) * (test[0] - temp[0]) + (test[1] - temp[1]) * (test[1] - temp[1])));
    }

    printf("start\n");
    KDTree<param> kdtree(test_datas, 2);
    printf("end\n");

    std::sort(len_list.begin(), len_list.end());
    int k = 5;
    while (len_list.size() > k)
    {
        len_list.pop_back();
    }
    display(len_list);

    printf("start\n");
    param ep(test);
    std::vector<param *> pd = kdtree.getKNearestPoints(&ep, k);
    for (auto &itr : pd)
    {
        display(std::sqrt((test[0] - itr->data[0]) * (test[0] - itr->data[0]) + (test[1] - itr->data[1]) * (test[1] - itr->data[1])));
    }
    printf("end\n");

    return 0;
}
```