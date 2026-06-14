/** * ASU - EEE 520 Project - 4 
 * Group 19 implementation of LRU IPV 
 * Authors : Krutyanjay Shinde , Vaishnavi , Veena
 * * Purpose :- Implements the LRU IPV replacement policy logic.
 */

// Include the parameters file
#include "params/LRUIPVRP.hh"

// Include the header file that defines the class
#include "mem/cache/replacement_policies/lru_ipv.hh"

#include <cassert>
#include <memory>
#include <stdio.h> // For printf debugging

/**
 * Constructor for the LRUIPVRP policy.
 *
 * This constructor sets up all the initial values needed for the policy.
 * It stores the associativity (numWays), loads the IPV vector used for
 * updating recency on hits, and sets the position where new blocks will be inserted.
 * It also makes sure the IPV table has enough entries and initializes a counter
 * that helps assign each block to the correct (set, way) during creation.
 */
LRUIPVRP::LRUIPVRP(const Params *p)
  : ReplacementPolicy::Base(*p) // Pass by reference
{
    // Get associativity from the parameters
    numWays = p->numWays;

    // Set the IPV vector for Group 19
    std::vector<int> baseIpv = 
        {0, 0, 1, 0, 3, 0, 4, 2, 1, 0, 5, 1, 0, 0, 8, 11};
    
    // Set the insertion position as per the vector
    insertionPos = 12;

    // Handle associativities > 16 by padding with 0s,
    int ipvSize = std::max(numWays, 16);
    ipvVector.resize(ipvSize, 0);
    
    // Copy the base IPV vector
    for (int i = 0; i < baseIpv.size() && i < ipvSize; ++i) {
        ipvVector[i] = baseIpv[i];
    }

    // Initialize the per-instance counter
    numEntries = 0;
}

/**
 * Destructor.
 */
LRUIPVRP::~LRUIPVRP()
{
}

/**
 * Called when a block is evicted or otherwise invalidated.
 *
 * This function updates the set’s recency ordering to reflect removal of a block.
 * It first notes the block’s current recency, shifts all strictly older blocks
 * one step toward MRU, and then places the invalidated block at LRU.
 * The victim becomes least-recently-used while preserving relative order.
 */
void
LRUIPVRP::invalidate(const std::shared_ptr<ReplacementPolicy::ReplacementData>& replacement_data) const
{


    // Cast the replacement data to our specific type
     std::shared_ptr<LRUIPVRPReplData> repl_data = std::static_pointer_cast<LRUIPVRPReplData>(replacement_data);
     std::shared_ptr<LRUIPVSharedState> shared_state = repl_data->sharedState;
     int wayID = repl_data->wayID;

    // Get the block's current recency
    int oldRecency = shared_state->recency[wayID];
    
    // The LRU position is always (numWays - 1)
    int lruPos = numWays - 1;

    // Demote this block:
    // Shift all blocks that were older than this one up by one
    for (int i = 0; i < numWays; ++i) {
        if (shared_state->recency[i] > oldRecency) {
            shared_state->recency[i]--;
        }
    }
    
    // Set the invalidated block's recency to the LRU position
    shared_state->recency[wayID] = lruPos;
}

/**
 * Called on a cache hit.
 *
 * This function performs the IPV promotion on a hit. It looks up the current
 * recency of the hit block, maps it to a new target recency via ipvVector, and
 * shifts any blocks that lie between [newRecency, oldRecency) one step older.
 * Finally, it assigns the hit block the IPV chosen recency, preserving total order.
 */
void
LRUIPVRP::touch(const std::shared_ptr<ReplacementPolicy::ReplacementData>& replacement_data) const
{

    // Cast the replacement data to our specific type
     std::shared_ptr<LRUIPVRPReplData> repl_data = std::static_pointer_cast<LRUIPVRPReplData>(replacement_data);
     std::shared_ptr<LRUIPVSharedState> shared_state = repl_data->sharedState;
     int wayID = repl_data->wayID;

    // get the old states
    std::vector<int> oldState = shared_state->recency;
    //printf("In touch.\n");
    //printf("\tSetID: %d\tindex: %d\n", repl_data->setID, wayID);
    //printf("\told sharedState: ");
    // for (int val : oldState) { printf("%d ", val); }


    // Get the block's current recency
    int oldRecency = shared_state->recency[wayID];
    
    // Get the new recency from the IPV vector
    int newRecency = ipvVector[oldRecency];

    // Promote this block:
    // Find all blocks with recency less than oldRecency and greater than or equal to newRecency
    // and increment their recency value.
    for (int i = 0; i < numWays; ++i) {
        if (i != wayID && shared_state->recency[i] >= newRecency && shared_state->recency[i] < oldRecency) {
            
            shared_state->recency[i]++;
        }
    }
    
    // Set the touched block's new recency
    shared_state->recency[wayID] = newRecency;


    // printf(" New sharedState is: ");
    // for (int val : shared_state->recency) { printf("%d ", val); }
    // printf("\n\n");
    // fflush(stdout);

}

/**
 * Called on a cache miss when a new block is inserted.
 *
 * This function places a just fetched block into the set according to IPV
 * insertionPos and shifts other blocks as needed to maintain a valid order.
 * Specifically, blocks whose recency is in [insertionPos, LRU) are demoted by 1,
 * and the new block takes insertionPos, this copies non-MRU insertion behavior.
 */
void LRUIPVRP::reset(const std::shared_ptr<ReplacementPolicy::ReplacementData>& replacement_data) const
{
    // Cast the replacement data to our specific type
    std::shared_ptr<LRUIPVRPReplData> repl_data = std::static_pointer_cast<LRUIPVRPReplData>(replacement_data);
    std::shared_ptr<LRUIPVSharedState> shared_state = repl_data->sharedState;
    // This wayID belongs to the victim, which is now being replaced
    int wayID = repl_data->wayID; 


    // Capture the old state 
    // std::vector<int> oldState = shared_state->recency;
    // printf("In reset.\n");
    // printf("\tSetID: %d\tindex: %d\n", repl_data->setID, wayID);
    // printf("\told sharedState: ");
    // for (int val : oldState) { printf("%d ", val); }


    // The new block is inserted at the insertion position
    int newRecency = insertionPos; 
    
    // The victim block's old recency was (numWays - 1)
    int oldRecency = numWays - 1; 

    // Demote blocks that are have lower recency than the new block's insertion position.
    for (int i = 0; i < numWays; ++i) {
        if (i != wayID && shared_state->recency[i] >= newRecency && shared_state->recency[i] < oldRecency) {

            shared_state->recency[i]++;
        }
    }
    
    // Set the new block's recency
    shared_state->recency[wayID] = newRecency;

    // printf(" New sharedState is: ");
    // for (int val : shared_state->recency) { printf("%d ", val); }
    // printf("\n\n");
    // fflush(stdout);

}

/**
 * Called on a cache miss to find a block to evict.
 *
 * This function scans the set’s recency array to find the way with the largest
 * recency value (the LRU position) and remembers its way index as victimWay.
 * It then walks the candidate entries to locate the matching wayID and returns
 * that ReplaceableEntry* to gem5. If not found for any reason, it returns index 0.
 */
ReplaceableEntry*
LRUIPVRP::getVictim(const ReplacementCandidates& candidates) const
{

    // Get the shared state from the first candidate
    std::shared_ptr<LRUIPVRPReplData> repl_data = std::static_pointer_cast<LRUIPVRPReplData>(
        candidates[0]->replacementData); 
    std::shared_ptr<LRUIPVSharedState> shared_state = repl_data->sharedState;


    // printf("In getVictim. SetID: %d\n", repl_data->setID);
    // printf("In getVictim. sharedState is: ");
    // for (int val : shared_state->recency) { printf("%d ", val); }


    // Find the wayID that has the highest recency value (LRU)
    // Initialize victimWay to -1 because valid way IDs are 0...(numWays-1).
    int victimWay = -1;
    int maxRecency = -1;
    for (int i = 0; i < numWays; ++i) {
        if (shared_state->recency[i] > maxRecency) {
            maxRecency = shared_state->recency[i];
            victimWay = i;
        }
    }

    // printf("    Victim: %d\n", victimWay);
    // fflush(stdout);


    // Find the candidate that corresponds to the victim's wayID
    for (ReplaceableEntry* candidate : candidates) {
        std::shared_ptr<LRUIPVRPReplData> rd = std::static_pointer_cast<LRUIPVRPReplData>(
            candidate->replacementData); 
        if (rd->wayID == victimWay) {
            return candidate;
        }
    }

    // return first candidate as default
    return candidates[0]; 
}

/**
 * Called by gem5 to create the replacement data for each block.
 *
 * This function allocates per block LRUIPVRPReplData and wires it to
 * its set’s shared recency state. It computes (set, way) based on a running
 * counter, creates the shared state object on the first way of each set, and
 * returns a shared_ptr<ReplacementData> that gem5 stores with the block.
 */
std::shared_ptr<ReplacementPolicy::ReplacementData>
LRUIPVRP::instantiateEntry()
{
    // Calculate the set and way for this new entry
    int set = numEntries / numWays;
    int way = numEntries % numWays;

    // If this is the first block (way 0) of a new set,
    // create a new shared state object for this set.
    if (way == 0) {
        setSharedStates.push_back(
            std::make_shared<LRUIPVSharedState>(numWays));
    }

    // Increment the total number of entries for this instance
    numEntries++;

    // Create the per-block replacement data, passing it a pointer
    // to its set's shared state, its wayID, and its setID.
    return std::shared_ptr<ReplacementPolicy::ReplacementData>(
        new LRUIPVRPReplData(setSharedStates[set], way, set));
}

/**
 * The create function is what the auto-generated 'params/LRUIPVRP.hh'
 * file calls to create an instance of the policy.
 *
 * It simply constructs and returns a new LRUIPVRP object, passing the
 * parameter bundle through to the base and to the policy constructor.
 * gem5 uses this hook to instantiate the policy from Python configs.
 */
LRUIPVRP*
LRUIPVRPParams::create() const
{
    return new LRUIPVRP(this);
}
