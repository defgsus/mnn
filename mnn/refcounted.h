/** @file

    @brief

    <p>(c) 2016, stefan.berke@modular-audio-graphics.com</p>
    <p>All rights reserved</p>

    <p>created 1/19/2016</p>
*/

#ifndef MNNSRC_REFCOUNTED_H
#define MNNSRC_REFCOUNTED_H

#include <atomic>

namespace MNN {


/** Basic strong reference counting type */
class RefCounted
{
public:

    RefCounted() : p_refcount_(1) { }

    /** Add reference to this object */
    void addRef() { ++p_refcount_; }

    /** Release reference to object.
        Destroys the object if referenceCount() goes to zero */
    void releaseRef() { if (--p_refcount_ == 0) delete this; }

    /** Returns the number of references on this object */
    int referenceCount() const { return p_refcount_; }

protected:
    virtual ~RefCounted() { }

private:

    std::atomic_int p_refcount_;
};


/** Deleter for smart-pointer of RefCounted types */
struct RefCountedDeleter
{
    void operator()(RefCounted * r) { r->releaseRef(); }
};


} // namespace MNN


#endif // MNNSRC_REFCOUNTED_H

