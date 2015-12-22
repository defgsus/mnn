/** @file

    @brief

    <p>(c) 2015, stefan.berke@modular-audio-graphics.com</p>
    <p>All rights reserved</p>

    <p>created 12/22/2015</p>
*/

#ifndef MNN_EXCEPTION_H
#define MNN_EXCEPTION_H

#include <exception>
#include <string>
#include <sstream>

namespace MNN {

#define MNN_EXCEPTION(text__) \
    throw ::MNN::Exception() << text__

class Exception
{
    std::string what_;
public:
    Exception() noexcept { }
    virtual ~Exception() noexcept { }

    virtual const char* what() const noexcept { return what_.c_str(); }

    template <typename T>
    Exception& operator << (T v)
    {
        std::stringstream s;
        s << v;
        what_ += s.str();
        return *this;
    }
};

} // namespace MNN

#endif // MNN_EXCEPTION_H

