#ifndef _KDTREE_H_
#define _KDTREE_H_

#include <algorithm>
#include <vector>
#include <eigen3/Eigen/Dense>
#include <iostream>
#include <chrono>

float calcDistance(Eigen::VectorXf *_p1, Eigen::VectorXf *_p2, int _dim);

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

protected:
    std::vector<float> k_nearest_dists;
    std::vector<T *> k_nearest_pts;
};

#endif // _KDTREE_H_