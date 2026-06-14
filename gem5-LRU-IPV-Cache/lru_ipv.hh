/** * ASU - EEE 520 Project - 4 
 * Group 19 implementation of LRU IPV 
 * Authors : Krutyanjay Shinde , Vaishnavi , Veena
 * * Purpose :- Defines class and member prototype for LRU IPV implementation
 */

/* Header Guard*/
#ifndef LRU_IPV_GUARD
#define LRU_IPV_GUARD

/* Include base replace policies class*/
#include <memory>
#include <vector>
#include "mem/cache/replacement_policies/base.hh"

/*
 * Forward declare the Params struct.
 */
struct LRUIPVRPParams;

/**
 * This class holds the shared recency state for ONE set.
 * All blocks in that set will have a shared_ptr to this object.
 */
class LRUIPVSharedState {
public:
    // This vector stores the recency value for each way
    std::vector<int> recency;

    LRUIPVSharedState(int numWays) : recency(numWays) {
        // Initialize recency values from 0 to numWays-1
        for (int i = 0; i < numWays; ++i) {
            recency[i] = i;
        }
    }
};

/**
 * This class holds the per-block replacement data.
 * It points to the shared state for its set.
 */
class LRUIPVRPReplData : public ReplacementPolicy::ReplacementData {
public:
    // A pointer to the shared state for this block's set
    std::shared_ptr<LRUIPVSharedState> sharedState;
    
    // This block's index within the set 
    int wayID;


    // The index of the set this block belongs to.
    int setID;

    LRUIPVRPReplData(std::shared_ptr<LRUIPVSharedState> _sharedState, int _wayID,
                   int _setID) 
        : sharedState(_sharedState), wayID(_wayID), setID(_setID) {} 
};


/*define our new class inherited from base replacement policy class*/
class LRUIPVRP : public ReplacementPolicy::Base
{
  private:
    // The Insertion/Promotion Vector (IPV)
    std::vector<int> ipvVector;
    
    // The position to insert new blocks
    int insertionPos;

    // This vector holds the shared state for ALL sets in the cache
    std::vector<std::shared_ptr<LRUIPVSharedState>> setSharedStates;
    
    // The associativity of the cache
    int numWays;

    // Counter must be a member variable, not static.
    int numEntries;

  public:

    /*define constructor prototype for class LRUIPVRP*/
    typedef LRUIPVRPParams Params;
    LRUIPVRP(const Params *p);

    /*define a destructor*/
    ~LRUIPVRP();

    /* Define our handler functions to handle LRUIPVRP operations */
      /**
      * Invalidate replacement data to set it as the next probable victim.
      * replacement_data Replacement data to be invalidated.
      */
    void invalidate(const std::shared_ptr<ReplacementPolicy::ReplacementData>& replacement_data)  const override;

    /**
      * Touch an entry to update its replacement data.
      * replacement_data Replacement data to be touched.
      */
    void touch(const std::shared_ptr<ReplacementPolicy::ReplacementData>& replacement_data) const override;

    /**
      * Reset replacement data. Used when an entry is inserted.
      * replacement_data Replacement data to be reset.
      */
    void reset(const std::shared_ptr<ReplacementPolicy::ReplacementData>& replacement_data) const override;

    /**
      * Find replacement victim using LRU timestamps.
      * Replacement entry to be replaced.
      */
    ReplaceableEntry* getVictim(const ReplacementCandidates& candidates) const override;

    /**
      * Instantiate a replacement data entry.
      * A shared pointer to the new replacement data.
      */
    std::shared_ptr<ReplacementPolicy::ReplacementData> instantiateEntry() override;
};

#endif // LRU_IPV_GUARD