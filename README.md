# KDTree
该KDTree的输入为std::vector\<T*\>, 其中T为任意 <span style="color:red">包含数据成员为 Eigen::VectorXf data</span> 的类
### 特点
1. 输入为对象指针的vector，返回值为对象的指针，可直接对返回对象进行修改,无需额外进行匹配
2. 针对不同问题，仅需声明不同的对象（包含data_）
3. 默认按照类中data_所有的数据来划分数据，可通过参数 *_split_dim* 指定KDTree根据data_的前几维进行划分

### 使用方法

输入对象
```cpp
class param
{
public:
    param(Eigen::VectorXf _data) : data_(_data) { ; }
    ~param() { ; }
    Eigen::VectorXf data_;
};
```

提供接口有
```cpp
void setData(std::vector<T *> _data_ptr_list, int _split_dims = -1);
void setData(std::vector<T> _data_ptr_list, int _split_dims = -1);
void insertNode(T *_param);
void removeNode(T *_param, bool _same_address = false);

T *getNearestPoint(T *_pt);
std::vector<T *> getPointsInRange(T *_pt, float _range);
std::vector<T *> getKNearestPoints(T *_pt, int _k);
```
目前删除节点使用的是替罪羊树， 失活节点数量记录在参数 *deactivate_nodes_num* 中

