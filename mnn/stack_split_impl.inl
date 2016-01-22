/** @file

    @brief

    <p>(c) 2016, stefan.berke@modular-audio-graphics.com</p>
    <p>All rights reserved</p>

    <p>created 1/21/2016</p>
*/


#define MNN_TEMPLATE template <typename Float>
#define MNN_STACKSPLIT StackSplit<Float>

MNN_TEMPLATE
MNN_STACKSPLIT::StackSplit()
{

}



MNN_TEMPLATE
StackSplit<Float>& MNN_STACKSPLIT::operator = (const Layer<Float>& layer)
{
    auto net = dynamic_cast<const StackSplit<Float>*>(&layer);
    if (!net)
        return *this;

    Stack<Float>::clearLayers();

    for (size_t i = 0; i < net->numLayers(); ++i)
    {
        auto l = net->layer(i)->getCopy();
        Stack<Float>::addLayer(l);
    }

    return *this;
}


// ---------------- io -------------------

MNN_TEMPLATE
void MNN_STACKSPLIT::serialize(std::ostream& s) const
{
    s << id();
    // version
    s << " " << 1;
    // dimension
    s << " " << Stack<Float>::numLayers() << " ";
    // each layer
    for (auto l : Stack<Float>::layer_)
    {
        s << "\n";
        l->serialize(s);
    }
}

MNN_TEMPLATE
void MNN_STACKSPLIT::deserialize(std::istream& s)
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
    if (numLay != Stack<Float>::numLayers())
        MNN_EXCEPTION("Number of layers does not match in " << name()
                      << ", expected " << Stack<Float>::numLayers() << ", found " << numLay);
    // each layer
    for (auto l : Stack<Float>::layer_)
        l->deserialize(s);

    updateLayers();
}



// ----------- nn interface --------------

MNN_TEMPLATE
void MNN_STACKSPLIT::resize(size_t numIn, size_t numOut)
{
    assert(!"resize() not implemented for StackSplit");
    (void)numIn;
    (void)numOut;
}

MNN_TEMPLATE
void MNN_STACKSPLIT::grow(size_t nrIn, size_t nrOut, Float randomDev)
{
    assert(!"grow() not implemented for StackSplit");
    (void)nrIn;
    (void)nrOut;
    (void)randomDev;
}

MNN_TEMPLATE
size_t MNN_STACKSPLIT::numIn() const
{
    return bufferIn_.size();
        //layer_.empty() ? 0 : layer_.front()->numIn();
}

MNN_TEMPLATE
size_t MNN_STACKSPLIT::numOut() const
{
    return bufferOut_.size();
}


// ----------- layer interface -----------

MNN_TEMPLATE
void MNN_STACKSPLIT::updateLayers()
{
    {	// clear buffer (completely)
        std::vector<Float> tmp;
        tmp.swap( bufferIn_ );
    }

    if (Stack<Float>::layer_.empty())
        return;

    size_t numIn = Stack<Float>::layer_.front()->numIn();
    size_t numOut = 0;
    for (auto l : Stack<Float>::layer_)
    {
        if (l->numIn() != numIn)
            l->resize(numIn, l->numOut());
        numOut += l->numOut();
    }

    bufferIn_.resize(numIn);
    bufferOut_.resize(numOut);
}


// ----------- propagation ---------------

MNN_TEMPLATE
void MNN_STACKSPLIT::fprop(const Float * input, Float * output)
{
    if (Stack<Float>::layer_.empty())
        return;

    auto b = &bufferOut_[0];
    for (auto l : Stack<Float>::layer_)
    {
        l->fprop(input, b);
        b += l->numOut();
    }

    // copy to caller
    for (auto i = bufferOut_.begin(); i != bufferOut_.end(); ++i, ++output)
        *output = *i;
}


MNN_TEMPLATE
void MNN_STACKSPLIT::bprop(const Float * error, Float * error_output,
                            Float global_learn_rate)
{
    if (Stack<Float>::layer_.empty())
        return;

    if (error_output)
    {
        // clear error for accumulation
        for (size_t i=0; i<numIn(); ++i)
            error_output[i] = Float(0);

        for (auto l : Stack<Float>::layer_)
        {
            l->bprop(error, &bufferIn_[0], global_learn_rate);
            error += l->numOut();

            for (size_t i=0; i<numIn(); ++i)
                error_output[i] += bufferIn_[i];
        }
    }
    else
    {
        for (auto l : Stack<Float>::layer_)
        {
            l->bprop(error, nullptr, global_learn_rate);
            error += l->numOut();
        }
    }
}


// ----------- info -----------------------

MNN_TEMPLATE
void MNN_STACKSPLIT::info(std::ostream &out, const std::string& pf) const
{
    out <<         pf << "name       : " << name()
        << "\n" << pf << "layout     : ";
    if (Stack<Float>::layer_.empty())
        out << "empty";
    else
    {
        out << Stack<Float>::layer_[0]->numIn() << " ->";
        for (size_t i=0; i<Stack<Float>::layer_.size(); ++i)
            out << " " << Stack<Float>::layer_[i]->numOut();
    }
    out << "\n" << pf << "parameters : " << Stack<Float>::numParameters()
        << "\n" << pf << "weights av : " << Stack<Float>::getWeightAverage()
        << "\n";
    size_t k = 1;
    for (auto l = Stack<Float>::layer_.begin(); l != Stack<Float>::layer_.end(); ++l, ++k)
    {
        out << pf << "-- parallel layer " << k << " --\n";
        (*l)->info(out, pf + "  ");
    }
}

MNN_TEMPLATE
void MNN_STACKSPLIT::dump(std::ostream &out) const
{
    size_t k = 0;
    for (auto l = Stack<Float>::layer_.begin(); l != Stack<Float>::layer_.end(); ++l, ++k)
    {
        out << "-- parallel layer " << k << " --\n";
        (*l)->dump(out);
    }
}






#undef MNN_TEMPLATE
#undef MNN_STACKSPLIT
