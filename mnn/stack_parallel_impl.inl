/** @file

    @brief

    <p>(c) 2016, stefan.berke@modular-audio-graphics.com</p>
    <p>All rights reserved</p>

    <p>created 1/12/2016</p>
*/


#define MNN_TEMPLATE template <typename Float>
#define MNN_STACKPARALLEL StackParallel<Float>

MNN_TEMPLATE
MNN_STACKPARALLEL::StackParallel()
{

}

MNN_TEMPLATE
MNN_STACKPARALLEL::~StackParallel()
{
    // free layers
    for (auto l : layer_)
        delete l;
}


// ---------------- io -------------------

MNN_TEMPLATE
void MNN_STACKPARALLEL::serialize(std::ostream& s) const
{
    s << id();
    // version
    s << " " << 1;
    // dimension
    s << " " << numLayer();
    // each layer
    for (auto l : layer_)
        l->serialize(s);
}

MNN_TEMPLATE
void MNN_STACKPARALLEL::deserialize(std::istream& s)
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

    // dimension
    size_t numLay;
    s >> numLay;
    if (numLay != numLayer())
        MNN_EXCEPTION("Number of layers does not match in " << name()
                      << ", expected " << numLayer() << ", found " << numLay);
    // each layer
    for (auto l : layer_)
        l->deserialize(s);

    resizeBuffers_();
}


// ----------- nn interface --------------

MNN_TEMPLATE
void MNN_STACKPARALLEL::resize(size_t numIn, size_t numOut)
{
    assert(!"resize() not implemented for StackParallel");
    (void)numIn;
    (void)numOut;
}

MNN_TEMPLATE
void MNN_STACKPARALLEL::grow(size_t nrIn, size_t nrOut, Float randomDev)
{
    assert(!"grow() not implemented for StackParallel");
    (void)nrIn;
    (void)nrOut;
    (void)randomDev;
}

MNN_TEMPLATE
size_t MNN_STACKPARALLEL::numIn() const
{
    size_t num = 0;
    for (auto l : layer_)
        num += l->numIn();
    return num;
}

MNN_TEMPLATE
size_t MNN_STACKPARALLEL::numOut() const
{
    return bufferOut_.size();
}


MNN_TEMPLATE
void MNN_STACKPARALLEL::brainwash(Float amp)
{
    for (auto l : layer_)
        l->brainwash(amp);
}

// ----------- layer interface -----------

MNN_TEMPLATE
size_t MNN_STACKPARALLEL::numLayer() const
{
    return layer_.size();
}

MNN_TEMPLATE
void MNN_STACKPARALLEL::add(Layer<Float> * layer)
{
    layer_.push_back(layer);
    resizeBuffers_();
}

MNN_TEMPLATE
void MNN_STACKPARALLEL::insert(size_t index, Layer<Float> * layer)
{
    if (index >= layer_.size())
        layer_.push_back(layer);
    else
        layer_.insert(layer_.begin() + index, layer);
    resizeBuffers_();
}

MNN_TEMPLATE
void MNN_STACKPARALLEL::resizeBuffers_()
{
    {	// clear buffer (completely)
        std::vector<Float> tmp;
        tmp.swap( bufferOut_ );
    }

    if (layer_.empty())
        return;

    size_t num = 0;
    for (auto l : layer_)
        num += l->numOut();

    bufferOut_.resize(num);
}


// ----------- propagation ---------------

MNN_TEMPLATE
void MNN_STACKPARALLEL::fprop(const Float * input, Float * output)
{
    if (layer_.empty())
        return;

    auto b = &bufferOut_[0];
    for (auto l : layer_)
    {
        l->fprop(input, b);
        input += l->numIn();
        b += l->numOut();
    }

    // copy to caller
    for (auto i = bufferOut_.begin(); i != bufferOut_.end(); ++i, ++output)
        *output = *i;
}


MNN_TEMPLATE
void MNN_STACKPARALLEL::bprop(const Float * error, Float * error_output,
                            Float global_learn_rate)
{
    if (layer_.empty())
        return;

    for (auto l : layer_)
    {
        l->bprop(error, error_output, global_learn_rate);
        if (error_output)
            error_output += l->numIn();
        error += l->numOut();
    }
}


// ----------- info -----------------------

MNN_TEMPLATE
Float MNN_STACKPARALLEL::getWeightAverage() const
{
    Float a = 0.;
    for (auto l : layer_)
        a += l->getWeightAverage();
    if (!layer_.empty())
        a /= layer_.size();
    return a;
}

MNN_TEMPLATE
void MNN_STACKPARALLEL::info(std::ostream &out) const
{
    out << "name : " << name() << "\n";
    size_t k = 1;
    for (auto l = layer_.begin(); l != layer_.end(); ++l, ++k)
    {
        out << "-- parallel layer " << k << " --\n";
        (*l)->info(out);
    }
}

MNN_TEMPLATE
void MNN_STACKPARALLEL::dump(std::ostream &out) const
{
    size_t k = 0;
    for (auto l = layer_.begin(); l != layer_.end(); ++l, ++k)
    {
        out << "-- parallel layer " << k << " --\n";
        (*l)->dump(out);
    }
}






#undef MNN_TEMPLATE
#undef MNN_STACKPARALLEL

