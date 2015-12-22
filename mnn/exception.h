/** @file

    @brief

    <p>(c) 2015, stefan.berke@modular-audio-graphics.com</p>
    <p>All rights reserved</p>

    <p>created 12/22/2015</p>
*/

#ifndef MNN_EXCEPTION_H
#define MNN_EXCEPTION_H

#include <exception>

namespace MNN {

#define MNN_EXCEPTION(text__) throw ::MNN::Exception(text__)

class Exception
{
    const char * what_;
public:
  Exception(const char* what) noexcept : what_(what) { }
  virtual ~Exception() noexcept { }

  virtual const char* what() const noexcept { return what_; }
};

} // namespace MNN

#endif // MNN_EXCEPTION_H

