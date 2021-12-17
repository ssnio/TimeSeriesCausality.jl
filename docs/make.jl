using Documenter, Literate
using PhaseSlopeIndex

EXAMPLE_DIR = joinpath(@__DIR__, "literate")
OUT_DIR = joinpath(@__DIR__, "src/generated")

## Use Literate.jl to generate docs and notebooks of examples
for example in readdir(EXAMPLE_DIR)
    EXAMPLE = joinpath(EXAMPLE_DIR, example)

    Literate.markdown(EXAMPLE, OUT_DIR; documenter=true) # markdown for Documenter.jl
    Literate.notebook(EXAMPLE, OUT_DIR) # .ipynb notebook
    Literate.script(EXAMPLE, OUT_DIR) # .jl script
end

## Build docs
makedocs(;
    sitename="PhaseSlopeIndex.jl",
    format=Documenter.HTML(),
    modules=[PhaseSlopeIndex],
    pages=["Home" => "index.md", "Examples" => "generated/examples.md"],
)

deploydocs(;
    repo="github.com/ssnio/PhaseSlopeIndex.jl.git", devbranch="master", branch="gh-pages"
)
