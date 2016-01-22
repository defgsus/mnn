/**	@file

	@brief Serial Stack header

	@author def.gsus-
	@version 2012/10/19 started
*/
#ifndef MNNSRC_STACK_SERIAL_H_INCLUDED
#define MNNSRC_STACK_SERIAL_H_INCLUDED

#include <cmath>
#include <vector>
#include <iostream>

#include "stack.h"

namespace MNN {

template <typename Float>
class StackSerial
        : public Stack<Float>
{

public:

    StackSerial();

    // ----------- copying -------------------

    virtual StackSerial<Float> * cloneClass() const override
        { return new StackSerial<Float>(); }

    virtual StackSerial<Float>& operator = (const Layer<Float>&) override;


	// ----------- nn interface --------------

    // ------- Stack interface ---------------

    virtual void updateLayers() override;

	// ------- propagation -------------------

    virtual void fprop(const Float * input, Float * output) override;

    virtual void bprop(const Float * error, Float * error_output = 0,
                       Float global_learn_rate = 1) override;

	// ------- info --------------------------

    static const char* static_id() { return "stack_serial"; }
    virtual const char * id() const override { return static_id(); }
    virtual const char * name() const override { return "StackSerial"; }
    virtual void info(std::ostream &out = std::cout,
                      const std::string& postFix = "") const override;
    virtual void dump(std::ostream &out = std::cout) const override;


    // ------------- io ---------------

    virtual void serialize(std::ostream&) const override;
    virtual void deserialize(std::istream&) override;

protected:

	/** intermediate buffers */
	std::vector<std::vector<Float> > buffer_;

};

#include "stack_serial_impl.inl"

} // namespace MNN

#endif // MNNSRC_STACK_SERIAL_H_INCLUDED
