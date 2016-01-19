/** @file

    @brief

    <p>(c) 2016, stefan.berke@modular-audio-graphics.com</p>
    <p>All rights reserved</p>

    <p>created 1/12/2016</p>
*/

#ifndef MNNSRC_STACK_PARALLEL_H
#define MNNSRC_STACK_PARALLEL_H


#include <cmath>
#include <cassert>
#include <vector>
#include <iostream>

#include "stack.h"

namespace MNN {

template <typename Float>
class StackParallel
        : public Stack<Float>
{
public:

    StackParallel();

    // ----------- copying -------------------

    virtual StackParallel<Float> * cloneClass() const override
        { return new StackParallel<Float>(); }

    virtual StackParallel<Float>& operator = (const Layer<Float>&) override;

    // ----------- nn interface --------------

    virtual void resize(size_t numIn, size_t numOut) override;
    virtual void grow(size_t nrIn, size_t nrOut, Float randomDev) override;

    virtual size_t numIn() const override;
    virtual size_t numOut() const override;

    // ------- layer interface ---------------

    virtual void updateLayers() override;

    // ------- propagation -------------------

    virtual void fprop(const Float * input, Float * output) override;

    virtual void bprop(const Float * error, Float * error_output = 0,
                       Float global_learn_rate = 1) override;

    // ------- info --------------------------

    static const char* static_id() { return "stack_parallel"; }
    virtual const char * id() const override { return static_id(); }
    virtual const char * name() const override { return "StackParallel"; }
    virtual void info(std::ostream &out = std::cout,
                      const std::string& postFix = "") const override;
    virtual void dump(std::ostream &out = std::cout) const override;

    // ------------- io ---------------

    virtual void serialize(std::ostream&) const override;
    virtual void deserialize(std::istream&) override;

protected:

    /** output buffer */
    std::vector<Float> bufferOut_;
};

#include "stack_parallel_impl.inl"

} // namespace MNN

#endif // MNNSRC_STACK_PARALLEL_H

