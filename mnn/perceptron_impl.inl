/**	@file perceptron_impl.inl

	@brief Perceptron implementation

	@author def.gsus-
	@version 2012/10/15 started
*/

#define MNN_TEMPLATE template <typename Float, class ActFunc>
#define MNN_PERCEPTRON Perceptron<Float, ActFunc>

MNN_TEMPLATE
MNN_PERCEPTRON::Perceptron(size_t nrIn, size_t nrOut, Float learnRate, bool bc)
    : learnRate_	(learnRate)
    , momentum_     (.1)
    , biasCell_     (bc)
{
	resize(nrIn, nrOut);
}

MNN_TEMPLATE
MNN_PERCEPTRON::~Perceptron()
{

}

// ---------------- io -------------------

MNN_TEMPLATE
void MNN_PERCEPTRON::serialize(std::ostream& s) const
{
    s << id();
    // version
    s << " " << 1;
    // settings
    s << " " << learnRate_ << " " << momentum_ << " " << biasCell_;
    // dimension
    s << " " << numIn() << " " << numOut();
    // weights
    for (auto w : weight_)
        s << " " << w;
}

MNN_TEMPLATE
void MNN_PERCEPTRON::deserialize(std::istream& s)
{
    std::string str;
    s >> str;
    if (str != id())
        MNN_EXCEPTION("Expected '" << id()
                      << "' in stream, found '" << str << "'");
    // version
    int ver;
    s >> ver;
    if (ver > 1)
        MNN_EXCEPTION("Wrong version in " << name());

    // settings
    s >> learnRate_ >> momentum_ >> biasCell_;
    // dimension
    size_t numIn, numOut;
    s >> numIn >> numOut;
    resize(numIn, numOut);
    // weights
    for (auto& w : weight_)
        s >> w;
}


// ----------- nn interface --------------

MNN_TEMPLATE
void MNN_PERCEPTRON::resize(size_t nrIn, size_t nrOut)
{
    if (biasCell_)
        ++nrIn;
    input_.resize(nrIn);
	output_.resize(nrOut);
    weight_.resize(nrIn * nrOut);
    prevDelta_.resize(nrIn * nrOut);
}

MNN_TEMPLATE
void MNN_PERCEPTRON::grow(size_t nrIn, size_t nrOut, Float randomDev)
{
    if (nrIn < numIn() || nrOut < numOut())
        return;

    if (biasCell_)
        ++nrIn;

    // copy weights
    std::vector<Float>
            weight(nrIn * nrOut);
    size_t o;
    for (o=0; o<output_.size(); ++o)
    {
        size_t i;
        for (i=0; i<input_.size(); ++i)
            weight[o * nrIn + i] = weight_[o * input_.size() + i];
        // choose random input to copy
        size_t ri = size_t(rand()) % input_.size();
        // run through additional inputs
        for (; i<nrIn; ++i)
            weight[o * nrIn + i] = weight_[o * input_.size() + ri]
                                    + rndg(Float(0), randomDev);
    }
    // run through additional outputs
    for (; o<nrOut; ++o)
    {
        // choose random input and output to copy
        size_t ro = size_t(rand()) % output_.size();
        size_t ri = size_t(rand()) % input_.size();

        size_t i;
        for (i=0; i<input_.size(); ++i)
            weight[o * nrIn + i] = weight_[ro * input_.size() + i];
        for (; i<nrIn; ++i)
            weight[o * nrIn + i] = weight_[ro * input_.size() + ri]
                                    + rndg(Float(0), randomDev);
    }
    // assign new weights
    weight_ = weight;
    // resize other buffers
    input_.resize(nrIn); for (auto&f : input_) f = 0;
    if (biasCell_) input_[input_.size()-1] = 1;
    output_.resize(nrOut); for (auto&f : output_) f = 0;
    prevDelta_.resize(nrIn * nrOut); for (auto&f : prevDelta_) f = 0;
}

MNN_TEMPLATE
size_t MNN_PERCEPTRON::numIn() const
{
    return biasCell_? input_.size() - 1 : input_.size();
}

MNN_TEMPLATE
size_t MNN_PERCEPTRON::numOut() const
{
	return output_.size();
}


MNN_TEMPLATE
void MNN_PERCEPTRON::brainwash(Float amp)
{
	// reset in/out
	for (auto e = input_.begin(); e != input_.end(); ++e)
		*e = 0.0;
	for (auto e = output_.begin(); e != output_.end(); ++e)
		*e = 0.0;

    if (input_.empty() || output_.empty())
        return;

    if (biasCell_)
        input_.back() = 1.;

	// randomize weights (assume normalized states)
    Float f = amp / std::sqrt(input_.size());
	for (auto e = weight_.begin(); e != weight_.end(); ++e)
		*e = rnd(-f, f);

    // reset momentum
    for (auto& f : prevDelta_)
        f = 0.;
}


// ----------- propagation ---------------

MNN_TEMPLATE
void MNN_PERCEPTRON::fprop(const Float * input, Float * output)
{
	// copy to internal data
    if (!biasCell_)
        for (auto i = input_.begin(); i != input_.end(); ++i, ++input)
            *i = *input;
    else
        for (size_t i=0; i<input_.size(); ++i, ++input)
            input_[i] = *input;

	// propagate
	auto w = weight_.begin();
	for (auto o = output_.begin(); o != output_.end(); ++o)
	{
		Float sum = 0;
		for (auto i = input_.begin(); i != input_.end(); ++i, ++w)
		{
			sum += *i * *w;
		}

		*o = ActFunc::activation(sum);
	}

	// copy to caller
    std::copy(output_.begin(), output_.end(), output);
}


MNN_TEMPLATE
void MNN_PERCEPTRON::bprop(const Float * error, Float * error_output,
                           Float global_learn_rate)
{
	global_learn_rate *= learnRate_;

    const Float * e;

	// pass error through
	if (error_output)
    for (size_t i = 0; i<numIn(); ++i, ++error_output)
	{
		Float sum = 0;
		e = error;
		for (size_t o = 0; o < output_.size(); ++o, ++e)
		{
			sum += *e * weight_[o * input_.size() + i];
		}
		*error_output = sum;
	}

	// backprob derivative
    auto w = &weight_[0];
    auto pd = &prevDelta_[0];
	e = error;
	for (auto o = output_.begin(); o != output_.end(); ++o, ++e)
	{
        Float de = ActFunc::derivative(*e, *o);

        for (auto i = input_.begin(); i != input_.end(); ++i, ++w, ++pd)
		{
            *pd = momentum_ * *pd
                + global_learn_rate * de * *i;
            *w += *pd;
		}

	}
}


// ----------- info -----------------------

MNN_TEMPLATE
Float MNN_PERCEPTRON::getWeightAverage() const
{
    Float a = 0.;
    for (auto w : weight_)
        a += std::abs(w);
    if (!weight_.empty())
        a /= weight_.size();
    return a;
}

MNN_TEMPLATE
void MNN_PERCEPTRON::info(std::ostream &out) const
{
	out <<   "name       : " << name()
        << "\nlearnrate  : " << learnRate_
        << "\nmomentum   : " << momentum_
        << "\ninputs     : " << numIn()
            << (biasCell_ ? " (+1 bias)" : "")
        << "\noutputs    : " << numOut()
		<< "\nactivation : " << ActFunc::static_name()
		<< "\n";
}

MNN_TEMPLATE
void MNN_PERCEPTRON::dump(std::ostream &out) const
{
	out << "inputs:";
	for (auto e = input_.begin(); e != input_.end(); ++e)
		out << " " << *e;

	out << "\noutputs:";
	for (auto e = output_.begin(); e != output_.end(); ++e)
		out << " " << *e;

	out << "\nweights:";
	for (auto e = weight_.begin(); e != weight_.end(); ++e)
		out << " " << *e;

	out << "\n";
}






#undef MNN_TEMPLATE
#undef MNN_PERCEPTRON
