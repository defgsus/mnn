/** @file

    @brief

    <p>(c) 2016, stefan.berke@modular-audio-graphics.com</p>
    <p>All rights reserved</p>

    <p>created 1/21/2016</p>
*/

#ifndef STACK_SPLIT_H
#define STACK_SPLIT_H

#include <cmath>
#include <vector>
#include <iostream>

#include "stack.h"

namespace MNN {

template <typename Float>
class StackSplit
        : public Stack<Float>
{

public:

    StackSplit();

    // ----------- copying -------------------

    virtual StackSplit<Float> * cloneClass() const override
        { return new StackSplit<Float>(); }

    virtual StackSplit<Float>& operator = (const Layer<Float>&) override;


    // ----------- nn interface --------------

    virtual void resize(size_t numIn, size_t numOut) override;
    virtual void grow(size_t nrIn, size_t nrOut, Float randomDev) override;

    virtual size_t numIn() const override;
    virtual size_t numOut() const override;

    // ------- Stack interface ---------------

    virtual void updateLayers() override;

    // ------- propagation -------------------

    virtual void fprop(const Float * input, Float * output) override;

    virtual void bprop(const Float * error, Float * error_output = 0,
                       Float global_learn_rate = 1) override;

    // ------- info --------------------------

    static const char* static_id() { return "stack_split"; }
    virtual const char * id() const override { return static_id(); }
    virtual const char * name() const override { return "StackSplit"; }
    virtual void info(std::ostream &out = std::cout,
                      const std::string& postFix = "") const override;
    virtual void dump(std::ostream &out = std::cout) const override;


    // ------------- io ---------------

    virtual void serialize(std::ostream&) const override;
    virtual void deserialize(std::istream&) override;

protected:

    std::vector<Float> bufferIn_, bufferOut_;

};

#include "stack_split_impl.inl"

} // namespace MNN


#endif // STACK_SPLIT_H

