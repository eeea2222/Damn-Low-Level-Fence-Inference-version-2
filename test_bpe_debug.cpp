#include <iostream>
#include <vector>
#include <string>
#include <unordered_map>

using namespace std;

class Tokenizer {
public:
    unordered_map<string, int> merges_;
    void bpe_merge(vector<string>& words) const {
        if (words.size() < 2) return;

        struct Symbol {
            int prev, next;
            int rank; // rank of pair (this, next)
        };

        vector<Symbol> syms(words.size());
        for (int i = 0; i < (int)words.size(); ++i) {
            syms[i].prev = i - 1;
            syms[i].next = i + 1;
            syms[i].rank = 1e9;
        }
        syms.back().next = -1;

        auto eval_pair = [&](int i) {
            if (i < 0 || i >= (int)syms.size() || syms[i].next < 0 || syms[i].next >= (int)syms.size()) {
                if(i >= 0 && i < (int)syms.size()) syms[i].rank = 1e9;
                return;
            }
            string pair = words[i] + " " + words[syms[i].next];
            auto it = merges_.find(pair);
            syms[i].rank = (it != merges_.end()) ? it->second : 1e9;
        };

        // Initial evaluation
        for (int i = 0; i < (int)words.size() - 1; ++i) {
            eval_pair(i);
        }

        while (true) {
            int best_rank = 1e9;
            int best_i = -1;
            
            // Find pair with lowest rank using index traversal
            int curr = 0;
            while (curr >= 0) {
                if (syms[curr].rank < best_rank) {
                    best_rank = syms[curr].rank;
                    best_i = curr;
                }
                curr = syms[curr].next;
            }

            if (best_i == -1 || best_rank == 1e9) break;

            // Merge best_i and best_i.next
            int right_i = syms[best_i].next;
            
            // Combine text in place
            words[best_i] += words[right_i];
            
            // Update linked list pointers
            syms[best_i].next = syms[right_i].next;
            if (syms[right_i].next >= 0) {
                syms[syms[right_i].next].prev = best_i;
            }

            // Re-evaluate affected pairs immediately bordering the merged pair
            eval_pair(syms[best_i].prev);
            eval_pair(best_i);
        }

        // Collect result
        vector<string> res;
        int curr = 0;
        while (curr >= 0) {
            res.push_back(std::move(words[curr]));
            curr = syms[curr].next;
        }
        words = std::move(res);
    }
};

int main() {
    Tokenizer t;
    t.merges_ = {{"a b", 1}, {"ab c", 2}, {"x y", 3}};
    vector<string> w = {"a", "b", "c", "x", "y", "z"};
    t.bpe_merge(w);
    for(auto s : w) cout << s << "|";
    cout << endl;
    return 0;
}
