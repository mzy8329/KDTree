#ifndef _KDTREE_H_
#define _KDTREE_H_

#include <algorithm>
#include <vector>
#include <eigen3/Eigen/Dense>
#include <iostream>
#include <chrono>

float calcDistance(const Eigen::VectorXf& _p1, const Eigen::VectorXf& _p2, int _dim);

template <typename T>
class Node
{
public:
    Node(T* _param, int _split_dim) : param(_param), split_dim_(_split_dim)
    {
        left = nullptr;
        right = nullptr;
        parent = nullptr;

        deactivate = false;
    }
    ~Node() { ; }

    T* param;
    int split_dim_;
    bool deactivate;

    Node<T>* left;
    Node<T>* right;
    Node<T>* parent;
};

template <typename T>
class KDTree
{
public:
    KDTree() { root_node_ = nullptr; }
    KDTree(std::vector<T*> _data_ptr_list, int _split_dims = -1);
    KDTree(std::vector<T> _data_list, int _split_dims = -1);
    ~KDTree();

    void setData(std::vector<T*> _data_ptr_list, int _split_dims = -1);
    void setData(std::vector<T> _data_ptr_list, int _split_dims = -1);
    void insertNode(T* _param);
    void removeNode(T* _param, bool _same_address = false);

    T* getNearestPoint(T* _pt);
    std::vector<T*> getPointsInRange(T* _pt, float _range);
    std::vector<T*> getKNearestPoints(T* _pt, int _k);

protected:
    void getPointsInRangeInBranch(T* _pt, float _range, Node<T>* _branch_node, Node<T>* _start_node = nullptr);
    void getKNearestPointsInBranch(T* _pt, int _k, Node<T>* _branch_node, Node<T>* _start_node = nullptr);

    Node<T>* findNearestLeaf(T* _pt, Node<T>* _branch_node, bool _same_address = false);
    Node<T>* findAnotherSuBranch(Node<T>* _parent, Node<T>* _sub_node);

    void clear();
    Node<T>* buildTree(std::vector<T*>* _data_list, Node<T>* _parent_node);
    void deleteBranch(Node<T>* _root_node);

    Eigen::VectorXf calcVarience(std::vector<T*>* _datas);
    bool compareByDim(const T* _a, const T* _b, int _dim);
    void sort_vector_list(std::vector<T*>* _datas, int splid_dim);

public:
    std::vector<T> data_list_;
    std::vector<T*> data_ptr_list_;
    int split_dim_;
    Node<T>* root_node_;
    int deactivate_nodes_num_;

protected:
    std::vector<float> k_nearest_dists_;
    std::vector<T*> k_nearest_pts_;
};

inline float calcDistance(const Eigen::VectorXf& _p1, const Eigen::VectorXf& _p2, int _dim)
{
    float temp = 0;
    for (int i = 0; i < _dim; i++)
    {
        temp += (_p1[i] - _p2[i]) * ((_p1)[i] - _p2[i]);
    }
    return std::sqrt(temp);
}

template <typename T>
KDTree<T>::KDTree(std::vector<T*> _data_ptr_list, int _split_dims)
{
    root_node_ = nullptr;
    setData(_split_dims, _data_ptr_list);
}

template <typename T>
KDTree<T>::KDTree(std::vector<T> _data_list, int _split_dims)
{
    root_node_ = nullptr;
    setData(_data_list, _split_dims);
}

template <typename T>
KDTree<T>::~KDTree()
{
    deleteBranch(root_node_);
}

// public -------------------- build tree --------------------//
template <typename T>
void KDTree<T>::setData(std::vector<T*> _data_ptr_list, int _split_dims)
{
    k_nearest_dists_.reserve(100);
    k_nearest_pts_.reserve(100);

    data_ptr_list_ = _data_ptr_list;
    clear();
    if (_data_ptr_list.size() <= 0)
    {
        return;
    }

    if (_split_dims == -1)
    {
        split_dim_ = _data_ptr_list[0]->data_.size();
    }
    else
    {
        split_dim_ = _split_dims;
    }
    deactivate_nodes_num_ = 0;

    root_node_ = buildTree(&_data_ptr_list, nullptr);
}

template <typename T>
void KDTree<T>::setData(std::vector<T> _data_list, int _split_dims)
{
    data_list_ = _data_list;
    data_ptr_list_.clear();
    for (auto& itr : data_list_)
    {
        data_ptr_list_.push_back(&itr);
    }
    setData(data_ptr_list_, _split_dims);
}

template <typename T>
void KDTree<T>::insertNode(T* _param)
{
    if (root_node_ == nullptr)
    {
        root_node_ = new Node<T>(_param, 0);
        return;
    }

    Node<T>* temp_node = root_node_;
    while (temp_node->left != nullptr || temp_node->right != nullptr)
    {
        if (_param->data_[temp_node->split_dim_] <= temp_node->param->data_[temp_node->split_dim_])
        {
            if (temp_node->left == nullptr)
            {
                break;
            }
            temp_node = temp_node->left;
        }
        else
        {
            if (temp_node->right == nullptr)
            {
                break;
            }
            temp_node = temp_node->right;
        }
    }

    Node<T>* node = new Node<T>(_param, temp_node->split_dim_);
    node->parent = temp_node;
    if (_param->data_[temp_node->split_dim_] <= temp_node->param->data_[temp_node->split_dim_])
    {
        temp_node->left = node;
    }
    else
    {
        temp_node->right = node;
    }
}

template <typename T>
void KDTree<T>::removeNode(T* _node, bool _same_address)
{
    Node<T>* temp_node = findNearestLeaf(_node, root_node_, _same_address);
    if (temp_node == nullptr)
    {
        return;
    }

    if (temp_node == root_node_)
    {
        if (root_node_->left == nullptr && root_node_->right == nullptr)
        {
            delete root_node_;
            root_node_ = nullptr;
            return;
        }
        else
        {
            root_node_->deactivate = true;
            deactivate_nodes_num_++;
        }
    }

    bool matched = false;
    if (_same_address)
    {
        matched = temp_node->param == _node;
    }
    else
    {
        matched = temp_node->param->data_ == _node->data_;
    }

    if (matched)
    {
        if (temp_node->left == nullptr && temp_node->right == nullptr)
        {
            if (temp_node->parent->left != nullptr)
            {
                if (temp_node == temp_node->parent->left)
                {
                    temp_node->parent->left = nullptr;
                }
                else
                {
                    temp_node->parent->right = nullptr;
                }
            }
            else
            {
                temp_node->parent->right = nullptr;
            }

            delete temp_node;
            temp_node = nullptr;
        }
        else
        {
            temp_node->deactivate = true;
            deactivate_nodes_num_++;
        }
    }
}

// public -------------------- find neighbor points --------------------//
template <typename T>
T* KDTree<T>::getNearestPoint(T* _pt)
{
    std::vector<T*> temp_ans = getKNearestPoints(_pt, 1);
    if (temp_ans.size() > 0)
    {
        return temp_ans[0];
    }
    else
    {
        return nullptr;
    }
}

template <typename T>
std::vector<T*> KDTree<T>::getPointsInRange(T* _pt, float _range)
{
    if (_range <= 0)
    {
        return {};
    }

    k_nearest_dists_.clear();
    k_nearest_pts_.clear();
    Node<T>* temp_node = findNearestLeaf(_pt, root_node_);
    if (_pt->data_ == temp_node->param->data_)
    {
        Node<T>* fit_node = temp_node;
        if (fit_node->left != nullptr)
        {
            Node<T>* left_leaf_node = findNearestLeaf(_pt, fit_node->left);
            getPointsInRangeInBranch(_pt, _range, left_leaf_node);
        }
        if (fit_node->right != nullptr)
        {
            Node<T>* right_leaf_node = findNearestLeaf(_pt, fit_node->right);
            getPointsInRangeInBranch(_pt, _range, right_leaf_node);
        }
        getPointsInRangeInBranch(_pt, _range, root_node_, fit_node->parent);
    }
    else
    {
        getPointsInRangeInBranch(_pt, _range, root_node_, temp_node);
    }
    return k_nearest_pts_;
}

template <typename T>
std::vector<T*> KDTree<T>::getKNearestPoints(T* _pt, int _k)
{
    if (_k > data_ptr_list_.size() || _k == 0 || data_ptr_list_.size() == 0)
    {
        return {};
    }

    k_nearest_dists_.clear();
    k_nearest_pts_.clear();
    Node<T>* temp_node = findNearestLeaf(_pt, root_node_);
    if (_pt->data_ == temp_node->param->data_ && !temp_node->deactivate)
    {
        Node<T>* fit_node = temp_node;
        if (fit_node->left != nullptr)
        {
            Node<T>* left_leaf_node = findNearestLeaf(_pt, fit_node->left);
            getKNearestPointsInBranch(_pt, _k, left_leaf_node);
        }
        if (fit_node->right != nullptr)
        {
            Node<T>* right_leaf_node = findNearestLeaf(_pt, fit_node->right);
            getKNearestPointsInBranch(_pt, _k, right_leaf_node);
        }
        getKNearestPointsInBranch(_pt, _k, root_node_, fit_node->parent);
    }
    else
    {
        getKNearestPointsInBranch(_pt, _k, root_node_, temp_node);
    }
    return k_nearest_pts_;
}

// protected -------------------- find neighbor points --------------------//
template <typename T>
void KDTree<T>::getPointsInRangeInBranch(T* _pt, float _range, Node<T>* _branch_node, Node<T>* _start_node)
{
    Node<T>* temp_node = _start_node;
    Node<T>* temp_node_last = nullptr;
    Node<T>* temp_node_another = nullptr;
    if (_start_node == nullptr)
    {
        temp_node = findNearestLeaf(_pt, _branch_node);
    }

    std::vector<Node<T>*> visited_node;
    float temp_dist;
    int temp_index;
    while (temp_node != _branch_node->parent)
    {
        if (std::find(visited_node.begin(), visited_node.end(), temp_node) != visited_node.end())
        {
            temp_node_last = temp_node;
            temp_node = temp_node->parent;
            continue;
        }

        if (!temp_node->deactivate)
        {
            temp_dist = calcDistance(temp_node->param->data_, _pt->data_, split_dim_);
            if (temp_dist <= _range)
            {
                k_nearest_pts_.push_back(temp_node->param);
                visited_node.push_back(temp_node);
            }
        }

        temp_node_another = findAnotherSuBranch(temp_node, temp_node_last);
        if (temp_node_another == nullptr || std::find(visited_node.begin(), visited_node.end(), temp_node_another) != visited_node.end())
        { // no another branch
            temp_node_last = temp_node;
            temp_node = temp_node->parent;
            continue;
        }

        if (abs(_pt->data_[temp_node->split_dim_] - temp_node->param->data_[temp_node->split_dim_]) < _range)
        {
            visited_node.push_back(temp_node);
            temp_node = findNearestLeaf(_pt, temp_node_another);
            temp_node_last = nullptr;
        }
        else
        {
            temp_node_last = temp_node;
            temp_node = temp_node->parent;
        }
    }
}

template <typename T>
void KDTree<T>::getKNearestPointsInBranch(T* _pt, int _k, Node<T>* _branch_node, Node<T>* _start_node)
{
    Node<T>* temp_node = _start_node;
    Node<T>* temp_node_last = nullptr;
    Node<T>* temp_node_another = nullptr;
    if (_start_node == nullptr)
    {
        temp_node = findNearestLeaf(_pt, _branch_node);
    }

    std::vector<Node<T>*> visited_node;
    float temp_dist;
    int temp_index;

    while (temp_node != _branch_node->parent)
    {
        if (std::find(visited_node.begin(), visited_node.end(), temp_node) != visited_node.end())
        {
            temp_node_last = temp_node;
            temp_node = temp_node->parent;
            continue;
        }

        if (!temp_node->deactivate)
        {
            temp_dist = calcDistance(temp_node->param->data_, _pt->data_, split_dim_);
            if (k_nearest_dists_.size() < _k)
            {
                k_nearest_dists_.push_back(temp_dist);
                k_nearest_pts_.push_back(temp_node->param);
                visited_node.push_back(temp_node);
            }
            else
            {
                temp_index = std::max_element(k_nearest_dists_.begin(), k_nearest_dists_.end()) - k_nearest_dists_.begin();
                if (temp_dist < k_nearest_dists_[temp_index])
                {
                    k_nearest_dists_.erase(k_nearest_dists_.begin() + temp_index);
                    k_nearest_pts_.erase(k_nearest_pts_.begin() + temp_index);

                    k_nearest_dists_.push_back(temp_dist);
                    k_nearest_pts_.push_back(temp_node->param);
                    visited_node.push_back(temp_node);
                }
            }
        }

        temp_node_another = findAnotherSuBranch(temp_node, temp_node_last);
        if (temp_node_another == nullptr || std::find(visited_node.begin(), visited_node.end(), temp_node_another) != visited_node.end())
        { // no another branch
            temp_node_last = temp_node;
            temp_node = temp_node->parent;
            continue;
        }

        temp_index = std::max_element(k_nearest_dists_.begin(), k_nearest_dists_.end()) - k_nearest_dists_.begin();
        if (k_nearest_dists_.size() < _k ||
            abs(_pt->data_[temp_node->split_dim_] - temp_node->param->data_[temp_node->split_dim_]) < k_nearest_dists_[temp_index])
        {
            visited_node.push_back(temp_node);
            temp_node = findNearestLeaf(_pt, temp_node_another);
            temp_node_last = nullptr;
        }
        else
        {
            temp_node_last = temp_node;
            temp_node = temp_node->parent;
        }
    }
}

// protected -------------------- find neighbor points --------------------//
template <typename T>
Node<T>* KDTree<T>::findNearestLeaf(T* _pt, Node<T>* _branch_node, bool _same_address)
{
    Node<T>* temp_node = _branch_node;
    if (_branch_node == nullptr)
    {
        return nullptr;
    }

    while (temp_node->left != nullptr || temp_node->right != nullptr)
    {
        if (_same_address)
        {
            if (_pt == temp_node->param)
            {
                if (!temp_node->deactivate)
                {
                    return temp_node;
                }
            }
        }
        else
        {
            if (_pt->data_ == temp_node->param->data_)
            {
                if (!temp_node->deactivate)
                {
                    return temp_node;
                }
            }
        }

        if (_pt->data_[temp_node->split_dim_] <= temp_node->param->data_[temp_node->split_dim_])
        {
            // temp_node = temp_node->left;
            temp_node = temp_node->left == nullptr ? temp_node->right : temp_node->left;
        }
        else
        {
            temp_node = temp_node->right == nullptr ? temp_node->left : temp_node->right;
        }
    }
    return temp_node;
}

template <typename T>
Node<T>* KDTree<T>::findAnotherSuBranch(Node<T>* _parent, Node<T>* _sub_node)
{
    if (_sub_node == _parent->left)
    {
        return _parent->right;
    }
    else if (_sub_node == _parent->right)
    {
        return _parent->left;
    }
    else
    {
        return nullptr;
    }
}

// protected -------------------- build tree --------------------//
template <typename T>
void KDTree<T>::clear()
{
    deleteBranch(root_node_);
}

template <typename T>
Node<T>* KDTree<T>::buildTree(std::vector<T*>* _data_ptr_list, Node<T>* _parent_node)
{
    int data_size = _data_ptr_list->size();
    if (data_size <= 0)
    {
        return nullptr;
    }
    if (data_size == 1)
    {
        Node<T>* node = new Node<T>(((*_data_ptr_list)[0]), 0);
        node->parent = _parent_node;
        return node;
    }

    Eigen::VectorXf varience = calcVarience(_data_ptr_list);
    if (std::abs(varience.size()) > 1e2)
    {
        return nullptr;
    }

    int s_dim = 0;
    varience.segment(0, split_dim_).maxCoeff(&s_dim);
    sort_vector_list(_data_ptr_list, s_dim);

    std::vector<T*> left_sub_tree;
    std::vector<T*> right_sub_tree;
    Node<T>* node = nullptr;

    if (data_size % 2 != 0)
    {
        for (int i = 0; i < data_size / 2; i++)
        {
            left_sub_tree.push_back((*_data_ptr_list)[i]);
            right_sub_tree.push_back((*_data_ptr_list)[data_size - 1 - i]);
        }
        node = new Node<T>((*_data_ptr_list)[data_size / 2], s_dim);
    }
    else
    {
        for (int i = 0; i < data_size / 2 - 1; i++)
        {
            left_sub_tree.push_back((*_data_ptr_list)[i]);
            right_sub_tree.push_back((*_data_ptr_list)[data_size - 1 - i]);
        }
        left_sub_tree.push_back((*_data_ptr_list)[data_size / 2 - 1]);
        node = new Node<T>((*_data_ptr_list)[data_size / 2], s_dim);
    }

    node->parent = _parent_node;
    node->left = buildTree(&left_sub_tree, node);
    node->right = buildTree(&right_sub_tree, node);
    return node;
}

template <typename T>
void KDTree<T>::deleteBranch(Node<T>* _root_node)
{
    if (_root_node == nullptr)
    {
        return;
    }

    Node<T>* temp_node = _root_node;
    while (_root_node->left != nullptr || _root_node->right != nullptr)
    {
        while (temp_node->left != nullptr)
        {
            temp_node = temp_node->left;
        }
        while (temp_node != _root_node && temp_node->right == nullptr)
        {
            if (temp_node == temp_node->parent->left)
            {
                temp_node = temp_node->parent;
                delete temp_node->left;
                temp_node->left = nullptr;
            }
            else
            {
                temp_node = temp_node->parent;
                delete temp_node->right;
                temp_node->right = nullptr;
            }
        }
        temp_node = temp_node->right;
    }

    if (_root_node->parent != nullptr)
    {
        temp_node = _root_node->parent;
        if (_root_node == temp_node->left)
        {
            delete _root_node;
            _root_node = nullptr;
            temp_node->left = nullptr;
        }
        else
        {
            delete _root_node;
            _root_node = nullptr;
            temp_node->right = nullptr;
        }
    }
    else
    {
        delete _root_node;
        root_node_ = nullptr;
    }
}

// protected -------------------- tools --------------------//
template <typename T>
Eigen::VectorXf KDTree<T>::calcVarience(std::vector<T*>* _data_ptr_list)
{
    int data_length = _data_ptr_list->size();
    int data_dim = (*(_data_ptr_list->begin()))->data_.size();

    Eigen::VectorXf mean(data_dim);
    mean.setZero();
    for (auto& itr : *_data_ptr_list)
    {
        mean += itr->data_;
    }
    mean = mean / float(data_length);

    Eigen::VectorXf varience(data_dim);
    varience.setZero();
    for (int i = 0; i < data_length; i++)
    {
        varience += ((*(_data_ptr_list->begin() + i))->data_ - mean).cwiseProduct((*(_data_ptr_list->begin() + i))->data_ - mean);
    }
    return varience / float(data_length);
}

template <typename T>
bool KDTree<T>::compareByDim(const T* _a, const T* _b, int _dim)
{
    return (_a->data_[_dim] < _b->data_[_dim]);
}

template <typename T>
void KDTree<T>::sort_vector_list(std::vector<T*>* _datas, int _split_dim)
{
    std::sort(_datas->begin(), _datas->end(),
        [this, _split_dim](const T* a, const T* b)
        {
            return compareByDim(a, b, _split_dim);
        });
}


#endif // _KDTREE_H_