#include "KDTree/KDTree.h"

float calcDistance(Eigen::VectorXf *_p1, Eigen::VectorXf *_p2, int _dim)
{
    float temp = 0;
    for (int i = 0; i < _dim; i++)
    {
        temp += ((*_p1)[i] - (*_p2)[i]) * ((*_p1)[i] - (*_p2)[i]);
    }
    return std::sqrt(temp);
}

template <typename T>
KDTree<T>::KDTree(int _split_dims, std::vector<T *> _data_ptr_list)
{
    root_node = nullptr;

    k_nearest_dists.reserve(100);
    k_nearest_pts.reserve(100);

    data_ptr_list = _data_ptr_list;
    setData(_split_dims, data_ptr_list);
}

template <typename T>
KDTree<T>::KDTree(int _split_dims, std::vector<T> _data_list)
{
    root_node = nullptr;

    k_nearest_dists.reserve(100);
    k_nearest_pts.reserve(100);

    data_list = _data_list;
    data_ptr_list.clear();
    for (auto &itr : data_list)
    {
        data_ptr_list.push_back(&itr);
    }
    setData(_split_dims, data_ptr_list);
}

template <typename T>
KDTree<T>::~KDTree()
{
    deleteBranch(root_node);
}

// public -------------------- build tree --------------------//
template <typename T>
void KDTree<T>::setData(int _split_dims, std::vector<T *> _data_ptr_list)
{
    clear();

    split_dim = _split_dims;
    deactivate_nodes_num = 0;

    k_nearest_dists.reserve(100);
    k_nearest_pts.reserve(100);

    root_node = buildTree(&_data_ptr_list, nullptr);
}

template <typename T>
void KDTree<T>::insertNode(T *_param)
{
    Node<T> *temp_node = root_node;
    while (temp_node->left != nullptr && temp_node->right != nullptr)
    {
        if (_param->data[temp_node->split_dim] <= temp_node->param->data[temp_node->split_dim])
        {
            temp_node = temp_node->right;
        }
        else
        {
            temp_node = temp_node->right;
        }
    }

    Node<T> *node = new Node<T>(_param, temp_node->split_dim);
    node->parent = temp_node;
    if (_param->data[temp_node->split_dim] <= temp_node->param->data[temp_node->split_dim])
    {
        temp_node->left = node;
    }
    else
    {
        temp_node->right = node;
    }
}

template <typename T>
void KDTree<T>::removeNode(T *_node)
{
    Node<T> *temp_node = findNearestLeaf(_node, root_node);
    if (temp_node->param->data == _node->data)
    {
        if (temp_node->left == nullptr && temp_node->right == nullptr && temp_node != root_node)
        {
            if (temp_node == temp_node->parent->left)
            {
                temp_node->parent->left = nullptr;
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
            deactivate_nodes_num++;
        }
    }
}

// public -------------------- find neighbor points --------------------//
template <typename T>
T *KDTree<T>::getNearestPoint(T *_pt)
{
    std::vector<T *> temp_ans = getPointsInRange(_pt, 9999, 1);
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
std::vector<T *> KDTree<T>::getPointsInRange(T *_pt, float _range, int _k)
{
    if (_range <= 0)
    {
        return {};
    }

    k_nearest_dists.clear();
    k_nearest_pts.clear();
    Node<T> *temp_node = findNearestLeaf(_pt, root_node);
    if (_pt->data == temp_node->param->data)
    {
        Node<T> *fit_node = temp_node;
        if (fit_node->left != nullptr)
        {
            Node<T> *left_leaf_node = findNearestLeaf(_pt, fit_node->left);
            getPointsInRangeInBranch(_pt, _range, left_leaf_node);
        }
        if (fit_node->right != nullptr)
        {
            Node<T> *right_leaf_node = findNearestLeaf(_pt, fit_node->right);
            getPointsInRangeInBranch(_pt, _range, right_leaf_node);
        }
        getPointsInRangeInBranch(_pt, _range, root_node, fit_node->parent);
    }
    else
    {
        getPointsInRangeInBranch(_pt, _range, root_node, temp_node);
    }
    return k_nearest_pts;
}

template <typename T>
std::vector<T *> KDTree<T>::getKNearestPoints(T *_pt, int _k)
{
    if (_k > data_ptr_list.size() || _k == 0 || data_ptr_list.size() == 0)
    {
        return {};
    }

    k_nearest_dists.clear();
    k_nearest_pts.clear();
    Node<T> *temp_node = findNearestLeaf(_pt, root_node);
    if (_pt->data == temp_node->param->data && !temp_node->deactivate)
    {
        Node<T> *fit_node = temp_node;
        if (fit_node->left != nullptr)
        {
            Node<T> *left_leaf_node = findNearestLeaf(_pt, fit_node->left);
            getKNearestPointsInBranch(_pt, _k, left_leaf_node);
        }
        if (fit_node->right != nullptr)
        {
            Node<T> *right_leaf_node = findNearestLeaf(_pt, fit_node->right);
            getKNearestPointsInBranch(_pt, _k, right_leaf_node);
        }
        getKNearestPointsInBranch(_pt, _k, root_node, fit_node->parent);
    }
    else
    {
        getKNearestPointsInBranch(_pt, _k, root_node, temp_node);
    }

    int max_index = std::max_element(k_nearest_dists.begin(), k_nearest_dists.end()) - k_nearest_dists.begin();
    T *temp = k_nearest_pts[max_index];
    k_nearest_pts.erase(k_nearest_pts.begin() + max_index);
    k_nearest_pts.push_back(temp);

    return k_nearest_pts;
}

// protected -------------------- find neighbor points --------------------//
template <typename T>
void KDTree<T>::getPointsInRangeInBranch(T *_pt, float _range, Node<T> *_branch_node, Node<T> *_start_node, int _k)
{
    T *temp_node = _start_node;
    T *temp_node_last = nullptr;
    T *temp_node_another = nullptr;
    if (_start_node == nullptr)
    {
        temp_node = findNearstLeaf(_pt, _branch_node);
    }

    std::vector<Node<T> *> visited_node;
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
            temp_dist = calcDistance(&temp_node->param->data, &_pt->data, split_dim);
            if (temp_dist <= _range)
            {
                k_nearest_pts.push_back(temp_node);
                visited_node.push_back(temp_node);
                if (_k > 0)
                {
                    if (k_nearest_pts.size() >= _k)
                    {
                        return;
                    }
                }
            }
        }

        temp_node_another = findAnotherSubranch(temp_node, temp_node_last);
        if (temp_node_another == nullptr || std::find(visited_node.begin(), visited_node.end(), temp_node_another) != visited_node.end())
        { // no another branch
            temp_node_last = temp_node;
            temp_node = temp_node->parent;
            continue;
        }

        if ((k_nearest_dists.size() < _k || abs(_pt->data[temp_node->split_dim] - temp_node->param->data[temp_node->split_dim]) < _range))
        {
            visited_node.push_back(temp_node);
            temp_node = findNearstLeaf(_pt, temp_node_another);
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
void KDTree<T>::getKNearestPointsInBranch(T *_pt, int _k, Node<T> *_branch_node, Node<T> *_start_node)
{
    Node<T> *temp_node = _start_node;
    Node<T> *temp_node_last = nullptr;
    Node<T> *temp_node_another = nullptr;
    if (_start_node == nullptr)
    {
        temp_node = findNearestLeaf(_pt, _branch_node);
    }

    std::vector<Node<T> *> visited_node;
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
            temp_dist = calcDistance(&temp_node->param->data, &_pt->data, split_dim);
            if (k_nearest_dists.size() < _k)
            {
                k_nearest_dists.push_back(temp_dist);
                k_nearest_pts.push_back(temp_node->param);
                visited_node.push_back(temp_node);
            }
            else
            {
                temp_index = std::max_element(k_nearest_dists.begin(), k_nearest_dists.end()) - k_nearest_dists.begin();
                if (temp_dist < k_nearest_dists[temp_index])
                {
                    k_nearest_dists.erase(k_nearest_dists.begin() + temp_index);
                    k_nearest_pts.erase(k_nearest_pts.begin() + temp_index);

                    k_nearest_dists.push_back(temp_dist);
                    k_nearest_pts.push_back(temp_node->param);
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

        temp_index = std::max_element(k_nearest_dists.begin(), k_nearest_dists.end()) - k_nearest_dists.begin();
        if (k_nearest_dists.size() < _k ||
            abs(_pt->data[temp_node->split_dim] - temp_node->param->data[temp_node->split_dim]) < k_nearest_dists[temp_index])
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
Node<T> *KDTree<T>::findNearestLeaf(T *_pt, Node<T> *_branch_node)
{
    Node<T> *temp_node = _branch_node;
    while (temp_node->left != nullptr || temp_node->right != nullptr)
    {
        if (_pt->data == temp_node->param->data)
        {
            if (!temp_node->deactivate)
            {
                return temp_node;
            }
        }

        if (_pt->data[temp_node->split_dim] <= temp_node->param->data[temp_node->split_dim])
        {
            temp_node = temp_node->left;
        }
        else
        {
            temp_node = temp_node->right == nullptr ? temp_node->left : temp_node->right;
        }
    }
    return temp_node;
}

template <typename T>
Node<T> *KDTree<T>::findAnotherSuBranch(Node<T> *_parent, Node<T> *_sub_node)
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
    deleteBranch(root_node);
}

template <typename T>
Node<T> *KDTree<T>::buildTree(std::vector<T *> *_data_ptr_list, Node<T> *_parent_node)
{
    int data_size = _data_ptr_list->size();
    if (data_size <= 0)
    {
        return nullptr;
    }
    if (data_size == 1)
    {
        Node<T> *node = new Node<T>(((*_data_ptr_list)[0]), 0);
        node->parent = _parent_node;
        return node;
    }
    Eigen::VectorXf varience = calcVarience(_data_ptr_list);

    int s_dim;
    varience.segment(0, split_dim).maxCoeff(&s_dim);
    sort_vector_list(_data_ptr_list, s_dim);

    std::vector<T *> left_sub_tree;
    std::vector<T *> right_sub_tree;
    Node<T> *node;
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
void KDTree<T>::deleteBranch(Node<T> *_root_node)
{
    if (_root_node == nullptr)
    {
        return;
    }

    Node<T> *temp_node = _root_node;
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
        root_node = nullptr;
    }
}

// protected -------------------- tools --------------------//
template <typename T>
Eigen::VectorXf KDTree<T>::calcVarience(std::vector<T *> *_data_ptr_list)
{
    int data_length = _data_ptr_list->size();
    int data_dim = (*(_data_ptr_list->begin()))->data.size();

    Eigen::VectorXf mean(data_dim);
    mean.setZero();
    for (auto &itr : *_data_ptr_list)
    {
        mean += itr->data;
    }
    mean = mean / float(data_length);

    Eigen::VectorXf varience(data_dim);
    varience.setZero();
    for (int i = 0; i < data_length; i++)
    {
        varience += ((*(_data_ptr_list->begin() + i))->data - mean).cwiseProduct((*(_data_ptr_list->begin() + i))->data - mean);
    }
    return varience / float(data_length);
}

template <typename T>
bool KDTree<T>::compareByDim(const T *_a, const T *_b, int _dim)
{
    return (_a->data[_dim] <= _b->data[_dim]);
}

template <typename T>
void KDTree<T>::sort_vector_list(std::vector<T *> *_datas, int _split_dim)
{
    std::sort(_datas->begin(), _datas->end(),
              [this, _split_dim](const T *a, const T *b)
              {
                  return compareByDim(a, b, _split_dim);
              });
}
