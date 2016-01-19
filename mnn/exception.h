/** @file

    @brief

    <p>(c) 2015, stefan.berke@modular-audio-graphics.com</p>
    <p>All rights reserved</p>

    <p>created 12/22/2015</p>
*/

#ifndef MNNSRC_EXCEPTION_H_INCLUDED
#define MNNSRC_EXCEPTION_H_INCLUDED

#include <exception>
#include <string>
#include <sstream>

namespace MNN {

#define MNN_EXCEPTION(text__) \
    throw ::MNN::Exception() << text__

/** Basic exception for all MNN classes.
    With stringstream support */
class Exception : public std::exception
{
    std::string what_;
public:
    Exception() noexcept { }
    virtual ~Exception() noexcept { }

    virtual const char* what() const noexcept { return what_.c_str(); }

    template <typename T>
    Exception& operator << (T v) noexcept
    {
        std::stringstream s;
        s << v;
        what_ += s.str();
        return *this;
    }
};

} // namespace MNN

#endif // MNNSRC_EXCEPTION_H_INCLUDED

