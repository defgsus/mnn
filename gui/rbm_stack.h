/** @file

    @brief

    <p>(c) 2015, stefan.berke@modular-audio-graphics.com</p>
    <p>All rights reserved</p>

    <p>created 12/25/2015</p>
*/

#ifndef RBM_STACK_H
#define RBM_STACK_H

#include <iomanip>

#include "mnn/mnn.h"


#ifndef LOG
#   define LOG(arg__) { std::cout << arg__ << std::endl; }
#endif


/*    / \    */
/*   /   \   */
/*  /.....\  */
/*     .     */
/*           */
template <typename Float, class Rbm>
class RbmStack
{
public:
    struct Sample
    {
        Sample() : err_cd(0.), err_rec(0.) { }
        std::vector<Float> data;
        Float err_cd, err_rec;
    };

private:
    std::vector<size_t> numCells_;
    std::vector<Rbm*> rbm_;
    std::vector<Sample*> samples_, higherSamples_;
    const size_t cdSteps_ = 50;
    const Float learnRate_ = 0.01;
    const Float momentum_ = .5;
public:

    Float err_min, err_max, err_sum;
    size_t epoch, err_count;

    RbmStack()
    {
        resetErrorStats();
    }

    void resetErrorStats()
    {
        err_min = -1.;
        err_max = 0.;
        err_sum = 0.;
        epoch = 0;
        err_count = 0;
    }

    const Rbm * rbm(size_t index) const { return rbm_[index]; }

    std::string statusString(size_t index) const
    {
        std::stringstream str;
        str << "step " << std::left << std::setw(9) << epoch
            << " err " << std::setw(9) << err_min
            << " - " << std::setw(9) << err_max
            << " av " << std::setw(9) << (err_sum / err_count)
            << " avweight " << rbm_[index]->getWeightAverage();
        return str.str();
    }

    const std::vector<Sample*>& samples() const { return samples_; }

    void clearSamples() { for (auto s : samples_) delete s; samples_.clear(); }
    void clearHigherSamples() { for (auto s : higherSamples_) delete s; higherSamples_.clear(); }

    size_t numIn() const { return numCells_.empty() ? 0 : numCells_[0]; }

    void setSize(const std::vector<size_t>& numCells)
    {
        numCells_ = numCells;
        for (size_t i=1; i<numCells.size(); ++i)
        {
            auto rbm = new Rbm(numCells[i-1], numCells[i]);
            rbm->brainwash();
            rbm->setMomentum(momentum_);
            rbm_.push_back(rbm);
        }
    }

    void addSample(const Float* s)
    {
        auto sam = new Sample;
        sam->data.resize(numIn());
        for (auto& f : sam->data)
            f = *s++;
        samples_.push_back(sam);
    }

    std::string layerFilename(size_t index) const
    {
        std::stringstream str;
        str << "rbm_layer_" << index << ".txt";
        return str.str();
    }

    void loadLayer(size_t index)
    {
        rbm_[index]->loadTextFile(layerFilename(index));
        std::cout << "loaded rbm layer " << index << " ("
                  << layerFilename(index) << ")\n";
        rbm_[index]->info();
    }

    /** Creates the net output for each sample_ into higherSamples_ */
    void createHigherSamples(size_t index)
    {
        LOG("creating output for each sample for layer " << index);
        clearHigherSamples();

        if (index == 0)
        for (size_t i = 0; i < samples_.size(); ++i)
        {
            // get space for output
            auto hsam = new Sample;
            higherSamples_.push_back(hsam);
            hsam->data.resize(rbm_[index]->numOut());
            // prob first layer
            rbm_[0]->fprop(&samples_[i]->data[0], &hsam->data[0]);
        }
        // XXX higher levels missing
        else abort();
    }

    void trainLayerLoop(size_t index, size_t maxEpoch = 300000)
    {
        LOG("\n------ TRAIN LAYER #" << index << " ------");

        resetErrorStats();

        Rbm* rbm = rbm_[index];
        while (epoch <= maxEpoch)
        {
            trainStep(index);

            if (epoch == 1 || (epoch % 1000) == 0)
            {
                LOG(statusString(index));

                err_max = 0.;
                err_min = -1.;
            }
        }

        rbm->saveTextFile(layerFilename(index));
        LOG("saved layer #" << index << " as '" << layerFilename(index) << "'");
    }

    void trainStep(size_t index)
    {
        Rbm* rbm = rbm_[index];

        // -- choose sample --

        Sample * sample;
        // real-data -> first layer
        if (index == 0)
        {
            size_t samIndex = size_t(rand()) % samples_.size();
            sample = samples_[samIndex];
        }
        // layer n-1 -> n
        else
        {
            size_t samIndex = size_t(rand()) % higherSamples_.size();
            sample = higherSamples_[samIndex];
        }

        // contrastive divergence training
        Float err = rbm->cd(&sample->data[0], cdSteps_, learnRate_);
        ++epoch;

        // -- gather error stats --

        sample->err_cd = err;

        if (err_min < 0.)
            err_min = err;
        else
            err_min = std::min(err_min, err);
        err_max = std::max(err_max, err);

        // CD error of 0. means probably no state at all
        if (err > 0.)
        {
            err_sum += err;
            ++err_count;
        }
    }
};

#endif // RBM_STACK_H

