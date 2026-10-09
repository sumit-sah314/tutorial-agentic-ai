# Blood Work Analysis Workflow

```mermaid
flowchart TD
    A[Blood report<br/>blood_work.txt] --> B[Load report into notebook]
    B --> C[Gemini 2.5 Flash]

    subgraph Stage 1: Extract and flag values
        C --> D[Extract every test value]
        D --> E[Compare each value with its<br/>report reference range]
        E --> F[Classify as HIGH, LOW,<br/>or NORMAL]
        F --> G[Structured blood-work analysis]
    end

    subgraph Stage 2: Summarize and recommend diet
        G --> H[Gemini 2.5 Flash]
        H --> I[Plain-language health summary]
        I --> J[Indian diet plan:<br/>foods to avoid and eat more of]
    end

    J --> K[Review by qualified healthcare professional]
    K --> L[Patient-facing guidance]
```

> **Important:** This workflow provides educational, AI-generated information only. A
> qualified healthcare professional must review the analysis before it is used for
> medical decisions.
