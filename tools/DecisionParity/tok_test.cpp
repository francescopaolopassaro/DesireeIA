// Prints the token ids of every string of a JSON array, one line per string.
#include "tokenizer/hf_bpe_tokenizer.h"
#include "models/decision_json.h"
#include <cstdio>
#include <fstream>
#include <iterator>
#include <string>

int main(int argc, char** argv) {
    if (argc < 3) { std::fprintf(stderr, "usage: tok_test tokenizer.json strings.json\n"); return 2; }
    desireeia::HfBpeTokenizer tok;
    if (!tok.load_file(argv[1])) { std::fprintf(stderr, "load failed\n"); return 1; }
    std::ifstream f(argv[2], std::ios::binary);
    std::string text((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    desireeia::djson::Value root;
    desireeia::djson::Parser p(text);
    if (!p.parse(root) || !root.is_arr()) { std::fprintf(stderr, "bad strings file\n"); return 1; }
    std::printf("special %d %d %d %d\n", tok.cls_id(), tok.sep_id(), tok.mask_id(), tok.pad_id());
    for (const auto& s : root.arr) {
        const auto ids = tok.encode(s.str);
        for (size_t i = 0; i < ids.size(); ++i) std::printf(i ? " %d" : "%d", ids[i]);
        std::printf("\n");
    }
    return 0;
}
