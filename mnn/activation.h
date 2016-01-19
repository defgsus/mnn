/**	@file

	@brief activation function objects

	@author def.gsus-
	@version 2012/10/15 started
*/
#ifndef MNN_ACTIVATION_H_INCLUDED
#define MNN_ACTIVATION_H_INCLUDED

#include <cmath>

namespace MNN {

namespace Activation {

/** base class for activation function objects */
struct Base
{
	virtual const char * name() const = 0;
};


/** linear activation */
struct Linear : public Base
{
	static const char * static_name() { return "linear"; }
	virtual const char * name() const { return static_name(); }

	template <typename Float>
	static Float activation(Float in) { return in; }

	template <typename Float>
	static Float derivative(Float error, Float /*state*/) { return error; }
};

/** max(0, x) */
struct LinearRectified : public Base
{
    static const char * static_name() { return "linear_rectified"; }
    virtual const char * name() const { return static_name(); }

    template <typename Float>
    static Float activation(Float in)
        { return std::max(Float(0), in); }

#if 1
    template <typename Float>
    static Float derivative(Float error, Float /*state*/)
        { return error; }
#else
    template <typename Float>
    static Float derivative(Float error, Float state)
        { return state <= Float(0) ? Float(0) : error; }
#endif
};


/** tangens hyperbolicus activation */
struct Tanh : public Base
{
	static const char * static_name() { return "tangens hyperbolicus"; }
	virtual const char * name() const { return static_name(); }

	template <typename Float>
    static Float activation(Float in) { return std::tanh(in); }

	template <typename Float>
    static Float derivative(Float error, Float state)
        { return (Float(1.) - state * state) * error; }
};

/** classical logistic activation */
struct Logistic : public Base
{
	static const char * static_name() { return "logistic"; }
	virtual const char * name() const { return static_name(); }

	template <typename Float>
    static Float activation(Float in)
        { return Float(1.)/(Float(1.) + std::exp(-in)); }

#if 1
	template <typename Float>
    static Float derivative(Float error, Float state)
        { return state * (Float(1.) - state) * error; }
#else
    template <typename Float>
    static Float derivative(Float error, Float state)
        { const Float e = std::exp(state);
            return (e / std::pow(Float(1) + e, Float(2))) * Float(2) * error; }
#endif
};

/** Logistic activation centered around 0.0 */
struct LogisticSymmetric : public Base
{
    static const char * static_name() { return "logistic_symmetric"; }
    virtual const char * name() const { return static_name(); }

    template <typename Float>
    static Float activation(Float in)
        { return Float(-1.) + Float(2.)/(Float(1.) + std::exp(-in)); }

    template <typename Float>
    static Float derivative(Float error, Float state)
        { return error * (Float(1.) - state * state); }
};

/** logistic10 activation */
struct Logistic10 : public Base
{
	static const char * static_name() { return "logistic10"; }
	virtual const char * name() const { return static_name(); }

	template <typename Float>
    static Float activation(Float in)
        { return Float(1.)/(Float(1.) + std::exp(Float(10.) * -in)); }

	template <typename Float>
    static Float derivative(Float error, Float state)
        { return state * (Float(1.) - state) * error * Float(10); }
};


/** sine activation */
struct Sine : public Base
{
    static const char * static_name() { return "sine"; }
    virtual const char * name() const { return static_name(); }

    template <typename Float>
    static Float activation(Float in) { return std::sin(in); }

    template <typename Float>
    static Float derivative(Float error, Float state)
        { return std::cos(state) * error; }
};

/** cosine activation */
struct Cosine : public Base
{
	static const char * static_name() { return "cosine"; }
	virtual const char * name() const { return static_name(); }

	template <typename Float>
    static Float activation(Float in) { return std::cos(in); }

	template <typename Float>
    static Float derivative(Float error, Float state)
        { return -std::sin(state) * error; }
};

/** smooth activation - a sigmoid curve */
struct Smooth : public Base
{
	static const char * static_name() { return "smooth"; }
	virtual const char * name() const { return static_name(); }

	template <typename Float>
    static Float activation(Float in)
        { return in*in*(Float(3.) - Float(2.) * in); }

	template <typename Float>
    static Float derivative(Float error, Float state)
        { return Float(6.)*(Float(1.)-state) * error; }
};

/** smooth2 activation - a steeper sigmoid curve */
struct Smooth2 : public Base
{
	static const char * static_name() { return "smooth2"; }
	virtual const char * name() const { return static_name(); }

	template <typename Float>
    static Float activation(Float in)
        { return in*in*in*(Float(6.)*in*in - Float(15.)*in + Float(10.)); }

	template <typename Float>
    static Float derivative(Float error, Float state)
        { return Float(30.) * std::pow(state-Float(1.), Float(2.))
                            * state * state * error; }
};


/** x > 0 ? 1 : 0 */
struct Threshold : public Base
{
    static const char * static_name() { return "threshold"; }
    virtual const char * name() const { return static_name(); }

    template <typename Float>
    static Float activation(Float in)
        { return (in > Float(0) ? Float(1) : Float(0)); }

    template <typename Float>
    static Float derivative(Float error, Float /*state*/)
        { return error; }
};


/** x > 0 ? 1 : -1 */
struct ThresholdSigned : public Base
{
    static const char * static_name() { return "threshold_signed"; }
    virtual const char * name() const { return static_name(); }

    template <typename Float>
    static Float activation(Float in)
        { return (in > Float(0) ? Float(1) : Float(-1)); }

    template <typename Float>
    static Float derivative(Float error, Float /*state*/)
        { return error; }
};


} // namespace Activation

} // namespace MNN





#endif // MNN_ACTIVATION_H_INCLUDED
