using Documenter, Literate
using TimeSeriesCausality

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
    sitename="TimeSeriesCausality.jl",
    format=Documenter.HTML(),
    modules=[TimeSeriesCausality],
    pages=["Home" => "index.md", "Examples" => "generated/examples.md"],
)

deploydocs(;
    repo="github.com/ssnio/TimeSeriesCausality.jl.git", devbranch="master", branch="gh-pages"
)
