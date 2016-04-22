#ifndef CAFFE_PYTHON_LAYER_HPP_
#define CAFFE_PYTHON_LAYER_HPP_

#include <boost/python.hpp>

#include <string>
#include <vector>

#include "caffe/layer.hpp"

namespace bp = boost::python;

namespace caffe {

#define PYTHON_LAYER_ERROR() { \
  PyObject *petype, *pevalue, *petrace; \
  PyErr_Fetch(&petype, &pevalue, &petrace); \
  bp::object etype(bp::handle<>(bp::borrowed(petype))); \
  bp::object evalue(bp::handle<>(bp::borrowed(bp::allow_null(pevalue)))); \
  bp::object etrace(bp::handle<>(bp::borrowed(bp::allow_null(petrace)))); \
  bp::object sio(bp::import("StringIO").attr("StringIO")()); \
  bp::import("traceback").attr("print_exception")( \
    etype, evalue, etrace, bp::object(), sio); \
  LOG(INFO) << bp::extract<string>(sio.attr("getvalue")())(); \
  PyErr_Restore(petype, pevalue, petrace); \
  throw; \
}

template <typename Dtype>
class PythonLayer : public Layer<Dtype> {
 public:
  PythonLayer(PyObject* self, const LayerParameter& param)
      : Layer<Dtype>(param), self_(bp::handle<>(bp::borrowed(self))) { }

  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
    self_.attr("param_str") = bp::str(
        this->layer_param_.python_param().param_str());
    self_.attr("setup")(bottom, top);
  }
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
    self_.attr("reshape")(bottom, top);
  }

  virtual inline bool ShareInParallel() const {
    return this->layer_param_.python_param().share_in_parallel();
  }

  virtual inline const char* type() const { return "Python"; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
    self_.attr("forward")(bottom, top);
  }
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    self_.attr("backward")(top, propagate_down, bottom);
  }

 private:
  bp::object self_;
};

}  // namespace caffe

#endif
