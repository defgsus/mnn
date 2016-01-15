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
    , dropOut_      (.5)
    , dropOutMode_  (DO_OFF)
    , biasCell_     (bc)
{
	resize(nrIn, nrOut);
}

MNN_TEMPLATE
MNN_PERCEPTRON::~Perceptron()
{

}

MNN_TEMPLATE
Perceptron<Float, ActFunc>& MNN_PERCEPTRON::operator = (const Layer<Float>& layer)
{
    auto net = dynamic_cast<const Perceptron<Float, ActFunc>*>(&layer);
    if (!net)
        return *this;

    input_ = net->input_;
    output_ = net->output_;
    weight_ = net->weight_;
    prevDelta_ = net->prevDelta_;
    drop_input_ = net->drop_input_;
    learnRate_ = net->learnRate_;
    momentum_ = net->momentum_;
    dropOut_ = net->dropOut_;
    dropOutMode_ = net->dropOutMode_;
    biasCell_ = net->biasCell_;

    return *this;
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
        for (size_t i=0; i<input_.size()-1; ++i, ++input)
            input_[i] = *input;

    if (dropOutMode_ == DO_TRAIN)
    {
        if (drop_input_.size() != input_.size())
            drop_input_.resize(input_.size());
        for (auto& d : drop_input_)
            d = rnd(Float(0), Float(1)) < dropOut_ ? 1 : 0;

        // propagate with dropout
        DenseMatrixDropout::fprop<Float, uint8_t, ActFunc>(
                    &input_[0], &output_[0], &weight_[0],
                    &drop_input_[0],
                    input_.size(), output_.size());
    }
    else if (dropOutMode_ == DO_PERFORM
             && drop_input_.size() == input_.size())
    {
        // propagate
        DenseMatrix::fprop_scale<Float, ActFunc>(
                    &input_[0], &output_[0], &weight_[0],
                    input_.size(), output_.size(),
                    Float(1) - dropOut_);
    }
    else
    {
        // propagate
        DenseMatrix::fprop<Float, ActFunc>(
                    &input_[0], &output_[0], &weight_[0],
                    input_.size(), output_.size());
    }

	// copy to caller
    std::copy(output_.begin(), output_.end(), output);
}


MNN_TEMPLATE
void MNN_PERCEPTRON::bprop(const Float * error, Float * error_output,
                           Float global_learn_rate)
{
    if (dropOutMode_ != DO_OFF && drop_input_.size() == input_.size())
    {
        // pass error through
        if (error_output)
            DenseMatrixDropout::bprop_stride<Float, uint8_t>(
                    error_output, error, &weight_[0],
                    &drop_input_[0],
                    numIn(), output_.size(), input_.size());

        // backprob derivative
        DenseMatrixDropout::gradient_descent<Float, uint8_t, ActFunc>(
                    &input_[0], &output_[0], error, &weight_[0], &prevDelta_[0],
                    &drop_input_[0],
                    input_.size(), output_.size(),
                    global_learn_rate * learnRate_,
                    momentum_);
    }
    else
    {
        // pass error through
        if (error_output)
            DenseMatrix::bprop_stride<Float>(
                    error_output, error, &weight_[0],
                    numIn(), output_.size(), input_.size());

        // backprob derivative
        DenseMatrix::gradient_descent<Float, ActFunc>(
                    &input_[0], &output_[0], error, &weight_[0], &prevDelta_[0],
                    input_.size(), output_.size(),
                    global_learn_rate * learnRate_,
                    momentum_);
    }
}

MNN_TEMPLATE
void MNN_PERCEPTRON::reconstruct(const Float* input, Float* reconstruction)
{
    if (!biasCell_)
    {
        // get code for input
        DenseMatrix::fprop<Float, ActFunc>(
                    input, &output_[0], &weight_[0],
                    input_.size(), output_.size());

        // get reconstruction from code
        DenseMatrix::fprop_transpose<Float, ActFunc>(
                    &output_[0], reconstruction, &weight_[0],
                    output_.size(), input_.size());
    }
    else
    {
        // copy to input
        for (size_t i=0; i<input_.size(); ++i, ++input)
            input_[i] = *input;

        // get code for input
        DenseMatrix::fprop<Float, ActFunc>(
                    &input_[0], &output_[0], &weight_[0],
                    input_.size(), output_.size());

        // get reconstruction space
        if (reconInput_.size() != input_.size())
            reconInput_.resize(input_.size());

        // get reconstruction from code
        DenseMatrix::fprop_transpose<Float, ActFunc>(
                    &output_[0], &reconInput_[0], &weight_[0],
                    output_.size(), input_.size());

        // copy to caller
        for (size_t i=0; i<input_.size()-1; ++i)
            reconstruction[i] = input_[i];
    }
}

MNN_TEMPLATE
Float MNN_PERCEPTRON::reconstructionTraining(
            const Float *dec_input, const Float* true_input,
            Float global_learn_rate)
{
    // copy to input
    if (!biasCell_)
        for (auto i = input_.begin(); i != input_.end(); ++i, ++dec_input)
            *i = *dec_input;
    else
        for (size_t i=0; i<input_.size()-1; ++i, ++dec_input)
            input_[i] = *dec_input;

    // get reconstruction space
    if (reconInput_.size() != input_.size())
        reconInput_.resize(input_.size());
    if (reconError_.size() != input_.size())
        reconError_.resize(input_.size());
    if (reconOutput_.size() != output_.size())
        reconOutput_.resize(output_.size());

    // get code for input
    DenseMatrix::fprop<Float, ActFunc>(
                &input_[0], &output_[0], &weight_[0],
                input_.size(), output_.size());

    // get reconstruction from code
    DenseMatrix::fprop_transpose<Float, Activation::Linear>(
                &output_[0], &reconInput_[0], &weight_[0],
                output_.size(), input_.size());

    // get reconstruction error
    Float err_sum = 0.;
    auto inp = true_input,
         err = &reconError_[0];
    for (auto i : reconInput_)
    {
        Float e = *inp++ - i;
        err_sum += std::abs(e);

        *err++ = e;//ActFunc::derivative(e, i);
    }
    err_sum /= input_.size();

    // find optimal code
    // (sum errors into code vector)
    DenseMatrix::fprop<Float, Activation::Linear>(
                &reconError_[0], &reconOutput_[0], &weight_[0],
                input_.size(), output_.size());

    // gradient descent using code error
    DenseMatrix::gradient_descent<Float, ActFunc>(
                &input_[0], &output_[0], &reconOutput_[0], &weight_[0], &prevDelta_[0],
                input_.size(), output_.size(),
                global_learn_rate * learnRate_,
                momentum_);

    return err_sum;
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
void MNN_PERCEPTRON::info(std::ostream &out, const std::string& pf) const
{
    out <<         pf << "name       : " << name()
        << "\n" << pf << "learnrate  : " << learnRate_
        << "\n" << pf << "momentum   : " << momentum_
        << "\n" << pf << "activation : " << ActFunc::static_name()
        << "\n" << pf << "inputs     : " << numIn()
            << (biasCell_ ? " (+1 bias)" : "")
        << "\n" << pf << "outputs    : " << numOut()
        << "\n" << pf << "parameters : " << numParameters()
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
