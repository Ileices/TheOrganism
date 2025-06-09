# RBY Singularity Framework - Unified Architecture Map

```ascii
                           ┌─────────────────────────────────────────┐
                           │          [RBY SINGULARITY]              │
                           │       Primary System Launch Script      │
                           │       UF+IO = R0.707 B0.500 Y0.793      │
                           └────────────────────┬────────────────────┘
                                                │
                                                ▼
                           ┌─────────────────────────────────────────┐
                           │   LAUNCH & INITIALIZATION ORCHESTRATOR  │
                           │ • Initialization Sequence               │
                           │ • Dependency Management                 │
                           │ • Process Spawning & Coordination       │
                           │ • Monitoring & Heartbeat                │
                           │ • Constants Configuration               │
                           │   ▹ Paths & Thresholds                  │
                           │   ▹ RBY & Fractal Settings             │
                           │   ▹ Resource & Evolution Settings       │
                           └────────────────────┬────────────────────┘
                                                │
             ┌─────────────────────┬───────────┴────────────┬────────────────────┬───────────────────┐
             │                     │                        │                    │                   │
             ▼                     ▼                        ▼                    ▼                   ▼
┌────────────────────────┐ ┌──────────────────┐ ┌────────────────────┐ ┌──────────────────┐ ┌──────────────────────┐
│  USER INTERACTION      │ │ NATURAL LANGUAGE │ │ LLM TRAINING &     │ │ CONTINUOUS      │ │ DECENTRALIZED HIGH-  │
│  INTERFACE            │ │ & GENERATIVE     │ │ EVOLUTION PIPELINE │ │ MONITORING &    │ │ PERFORMANCE COMPUTE  │
│ • Terminal Console    │ │ RESPONSE ENGINE  │ │ • Data Ingestion   │ │ SELF-MODIFYING │ │ NETWORK              │
│ • Graphical UI        │ │ • Language Core  │ │ • Fine-Tuning      │ │ MANAGER        │ │ • Volunteer Nodes    │
│ • Input Parser        │ │ • Code Generator │ │ • Model Compress.  │ │ • Behavior     │ │ • Task Distributor   │
│ • Output Renderer     │ │ • QA Module      │ │ • Seed Evolution   │ │   Analyzer     │ │ • Result Aggregator  │
│                       │ │                  │ │                    │ │ • Resource     │ │ • RAM/CPU Monitoring │
│                       │ │                  │ │                    │ │   Monitor      │ │   (psutil)           │
└──────────┬─────────────┘ └────────┬─────────┘ └────────┬───────────┘ └───────┬──────────┘ └─────────┬────────────┘
           │                        │                    │                     │                      │
           │                        │                    │      ┌──────────────┘                      │
           │                        │                    │      │                                     │
           │                        │                    │      ▼                                     │
           │                        │                    │   ┌────────────────────────────────┐       │
           │                        │                    │   │ RBY COMPONENT SYSTEM           │       │
           │                        │                    │   │ • ComponentRole Classification │       │
           │                        │                    │   │   ▹ RED (Perception)          │       │
           │                        │                    │   │   ▹ BLUE (Processing)         │       │
           │                        │                    │   │   ▹ YELLOW (Generation)       │       │
           │                        │                    │   │ • Integration Levels          │       │
           │                        │                    │   │   ▹ LOOSE/MEDIUM/TIGHT        │       │
           │                        │                    │   │ • Component Discovery         │       │
           │                        │                    │   │ • Code Analysis Engine       │       │
           │                        │                    │   │ • Integration Mapping        │       │
           │                        │                    │   └───────────────┬──────────────┘       │
           │                        │                    │                   │                      │
           │                        │                    │                   ▼                      │
           │                        │                    │   ┌────────────────────────────────┐     │
           │                        │                    │   │ 🧠 COMPONENT EVOLUTION         │     │
           │                        │                    │   │ • __init__()                   │     │
           │                        │                    │   │   ▹ purpose: Initialize        │     │
           │                        │                    │   │     evolution system           │     │
           │                        │                    │   │   ▹ called_by: processor       │     │
           │                        │                    │   │   ▹ depends_on: threading,     │     │
           │                        │                    │   │     singularity                │     │
           │                        │                    │   │ • load_history()              │     │
           │                        │                    │   │   ▹ purpose: Load optimization │     │
           │                        │                    │   │     history                   │     │
           │                        │                    │   │ • save_history()              │     │
           │                        │                    │   │   ▹ purpose: Save history     │     │
           │                        │                    │   │     to JSON                   │     │
           │                        │                    │   │ • start_evolution_console()   │     │
           │                        │                    │   │   ▹ purpose: Interactive      │     │
           │                        │                    │   │     evolution console         │     │
           │                        │                    │   │ • _scan_codebase()           │     │
           │                        │                    │   │   ▹ purpose: Find functions   │     │
           │                        │                    │   │     to optimize              │     │
           │                        │                    │   │   ▹ depends_on: ast          │     │
           │                        │                    │   │ • evolve_function()          │     │
           │                        │                    │   │   ▹ purpose: Evolve/optimize  │     │
           │                        │                    │   │     a specific function      │     │
           │                        │                    │   │   ▹ outputs_to: codebase     │     │
           │                        │                    │   │ • _locate_function()         │     │
           │                        │                    │   │   ▹ purpose: Find function   │     │
           │                        │                    │   │     in codebase              │     │
           │                        │                    │   │ • _extract_function_code()   │     │
           │                        │                    │   │   ▹ purpose: Extract code    │     │
           │                        │                    │   │     from file                │     │
           │                        │                    │   │   ▹ depends_on: ast          │     │
           │                        │                    │   │ • _generate_improvements()   │     │
           │                        │                    │   │   ▹ purpose: Create code     │     │
           │                        │                    │   │     mutations               │     │
           │                        │                    │   │   ▹ depends_on: ast          │     │
           │                        │                    │   │   ▹ outputs_to: improvements │     │
           │                        │                    │   │     list                     │     │
           │                        │                    │   │ • _optimize_loops()          │     │
           │                        │                    │   │   ▹ purpose: Optimize loop   │     │
           │                        │                    │   │     structures using AST     │     │
           │                        │                    │   │   ▹ depends_on: ast          │     │
           │                        │                    │   │   ▹ called_by: _generate_    │     │
           │                        │                    │   │     improvements             │     │
           │                        │                    │   │ • _combine_operations()      │     │
           │                        │                    │   │   ▹ purpose: Combine redund. │     │
           │                        │                    │   │     operations with AST      │     │
           │                        │                    │   │   ▹ called_by: _generate_    │     │
           │                        │                    │   │     improvements             │     │
           │                        │                    │   │ • _add_caching()             │     │
           │                        │                    │   │   ▹ purpose: Add caching for │     │
           │                        │                    │   │     repeated calculations    │     │
           │                        │                    │   │   ▹ called_by: _generate_    │     │
           │                        │                    │   │     improvements             │     │
           │                        │                    │   │ • _optimize_perception()     │     │
           │                        │                    │   │   ▹ purpose: Optimize Red    │     │
           │                        │                    │   │     components for input     │     │
           │                        │                    │   │   ▹ called_by: _generate_    │     │
           │                        │                    │   │     improvements             │     │
           │                        │                    │   │   ▹ depends_on: ast          │     │
           │                        │                    │   │   ▹ outputs_to: enhanced     │     │
           │                        │                    │   │     error handling          │     │
           │                        │                    │   │ • _optimize_processing()     │     │
           │                        │                    │   │   ▹ purpose: Optimize Blue   │     │
           │                        │                    │   │     components for compute   │     │
           │                        │                    │   │   ▹ called_by: _generate_    │     │
           │                        │                    │   │     improvements             │     │
           │                        │                    │   │   ▹ depends_on: ast, time    │     │
           │                        │                    │   │   ▹ outputs_to: performance  │     │
           │                        │                    │   │     tracking code           │     │
           │                        │                    │   │ • _optimize_generation()     │     │
           │                        │                    │   │   ▹ purpose: Optimize Yellow │     │
           │                        │                    │   │     components for creation  │     │
           │                        │                    │   │   ▹ called_by: _generate_    │     │
           │                        │                    │   │     improvements             │     │
           │                        │                    │   │   ▹ depends_on: ast          │     │
           │                        │                    │   │   ▹ outputs_to: quality      │     │
           │                        │                    │   │     validation code         │     │
           │                        │                    │   │ • _display_evolution_status()│     │
           │                        │                    │   │   ▹ purpose: Show evolution  │     │
           │                        │                    │   │     metrics & status        │     │
           │                        │                    │   │   ▹ called_by: evolution_    │     │
           │                        │                    │   │     console                  │     │
           │                        │                    │   │ • _display_detailed_status() │     │
           │                        │                    │   │   ▹ purpose: Show complete   │     │
           │                        │                    │   │     evolution metrics       │     │
           │                        │                    │   │   ▹ called_by: evolution_    │     │
           │                        │                    │   │     console                  │     │
           │                        │                    │   │ • _display_function_history()│     │
           │                        │                    │   │   ▹ purpose: Show history    │     │
           │                        │                    │   │     for a function          │     │
           │                        │                    │   │   ▹ called_by: evolution_    │     │
           │                        │                    │   │     console                  │     │
           │                        │                    │   │ • _display_help()           │     │
           │                        │                    │   │   ▹ purpose: Show evolution  │     │
           │                        │                    │   │     console commands        │     │
           │                        │                    │   │   ▹ called_by: evolution_    │     │
           │                        │                    │   │     console                  │     │
           │                        │                    │   │ • _test_improvements()       │     │
           │                        │                    │   │   ▹ purpose: Test mutations  │     │
           │                        │                    │   │     in sandbox               │     │
           │                        │                    │   │   ▹ depends_on: _benchmark_  │     │
           │                        │                    │   │     function                 │     │
           │                        │                    │   │   ▹ outputs_to: best_version │     │
           │                        │                    │   │ • _benchmark_function()      │     │
           │                        │                    │   │   ▹ purpose: Measure function│     │
           │                        │                    │   │     performance              │     │
           │                        │                    │   │   ▹ called_by: _test_        │     │
           │                        │                    │   │     improvements             │     │
           │                        │                    │   │   ▹ depends_on: _generate_   │     │
           │                        │                    │   │     test_cases, _setup_test_ │     │
           │                        │                    │   │     environment              │     │
           │                        │                    │   │ • _generate_test_cases()     │     │
           │                        │                    │   │   ▹ purpose: Create test data│     │
           │                        │                    │   │     based on function sig    │     │
           │                        │                    │   │   ▹ called_by: _benchmark_   │     │
           │                        │                    │   │     function                 │     │
           │                        │                    │   │ • _setup_test_environment() │     │
           │                        │                    │   │   ▹ purpose: Create isolated │     │
           │                        │                    │   │     test environment         │     │
           │                        │                    │   │   ▹ called_by: _benchmark_   │     │
           │                        │                    │   │     function                 │     │
           │                        │                    │   │   ▹ depends_on: importlib    │     │
           │                        │                    │   │ • _run_test()               │     │
           │                        │                    │   │   ▹ purpose: Execute function│     │
           │                        │                    │   │     with test arguments      │     │
           │                        │                    │   │   ▹ called_by: _benchmark_   │     │
           │                        │                    │   │     function                 │     │
           │                        │                    │   │ • _apply_improvement()       │     │
           │                        │                    │   │   ▹ purpose: Apply best      │     │
           │                        │                    │   │     mutation to codebase     │     │
           │                        │                    │   │   ▹ depends_on: ast,         │     │
           │                        │                    │   │     p2p_version_manager      │     │
           │                        │                    │   │ • start_background_evolution()│    │
           │                        │                    │   │   ▹ purpose: Init background │     │
           │                        │                    │   │     evolution thread        │     │
           │                        │                    │   │   ▹ called_by: launch.py     │     │
           │                        │                    │   │   ▹ depends_on: threading,   │     │
           │                        │                    │   │     time, random            │     │
           │                        │                    │   │   ▹ outputs_to: evolution_   │     │
           │                        │                    │   │     thread                   │     │
           │                        │                    │   └───────────────┬──────────────┘     │
           │                        │                    │                   │                    │
           └────────────────────────┼────────────────────┼───────────────────┼────────────────────┘
                                    │                    │                   │
                                    ▼                    │                   │
                        ┌────────────────────────────┐   │                   │
                        │    EXPANSION PHASE (C-AE)  │   │                   │
                        │  Crystallized Absolute     │   │                   │
                        │  Existence                 │   │                   │
                        │                            │   │                   │
                        │ ┌─────────────────────────┐│   │                   │
                        │ │   MULTIMODAL MEDIA      ││   │                   │
                        │ │   GENERATION UNIT       ││   │                   │
                        │ │ • Image Generator       ││   │                   │
                        │ │ • Video & Animation     ││◄──┘                   │
                        │ │ • 3D Asset Generator    ││                       │
                        │ │ • Audio & Speech        ││                       │
                        │ │ • Multimodal Orchestrator││                      │
                        │ └─────────────────────────┘│                       │
                        │                            │                       │
                        │ ┌─────────────────────────┐│                       │
                        │ │   CRYSTALLIZED AE CLASS ││                       │
                        │ │ • start_monitoring()    ││                       │
                        │ │ • stop_monitoring()     ││                       │
                        │ │ • _monitor_system_      ││                       │
                        │ │   resources()           ││                       │
                        │ │ • attempt_evolution()   ││◄──────────────────────┘
                        │ │ • trigger_compression() ││
                        │ │ • process_compression_  ││
                        │ │   queue()               ││
                        │ │ • compress_to_glyphs()  ││
                        │ │ • compress_all_and_     ││
                        │ │   reset()               ││
                        │ └─────────────────────────┘│
                        │                            │
                        │ ┌─────────────────────────┐│
                        │ │   RBY MEMORY MANAGEMENT ││
                        │ │ • record_excretion()    ││
                        │ │ • _calculate_dominant_  ││
                        │ │   rby()                 ││
                        │ │ • _save_excretion_to_   ││
                        │ │   disk()                ││
                        │ └─────────────────────────┘│
                        └─────────────┬──────────────┘
                                      │
                                      ▼
                    ┌─────────────────────────────────────────┐
                    │              IC-AE ZONE                 │
                    │        (Infected C-AE Sandboxes)        │
                    │                                         │
      ┌─────────────┼─────────────────┐   ┌─────────────────┼─────────────────┐
      │             │                 │   │                 │                 │
      ▼             │                 ▼   ▼                 │                 ▼
┌────────────────┐  │  ┌────────────────────────────────┐  │  ┌────────────────┐
│ INFECTED C-AE  │  │  │     INFECTED C-AE SANDBOX 2    │  │  │  INFECTED C-AE  │
│ SANDBOX 1      │◄─┼──┼────────────────────────────────┼──┼─►│  SANDBOX 3      │
│                │  │  │                                │  │  │                 │
│ • infect_script()│  │  │ • InfectedCAE Class         │  │  │                 │
│ • _evolve_      │  │  │ • record_excretion()         │  │  │                 │
│   infected_     │  │  │ • _save_excretion_to_disk()  │  │  │                 │
│   script()      │  │  │ • infect_script()            │  │  │                 │
└────────┬───────┘  │  └──────────────┬─────────────────┘  │  └─────────┬──────┘
         │          │                 │                    │            │
         ▼          │                 ▼                    │            ▼ 
┌────────────────┐  │  ┌────────────────────────────────┐  │  ┌────────────────┐
│  IIC-AE FRACTAL│◄─┼──┼────────────────────────────────┼──┼─►│  IIC-AE FRACTAL│
│(NESTED SANDBOX)│  │  │         IIC-AE FRACTAL         │  │  │(NESTED SANDBOX)│
└────────────────┘  │  └────────────────────────────────┘  │  └────────────────┘
                    │                                      │
                    └──────────────────┬──────────────────┬┘
                                       │                  │
                                       │                  │
                                       ▼                  │
                         ┌─────────────────────────┐      │
                         │       ABSULARITY        │      │
                         │    Maximum Expansion    │      │
                         │  (90% Storage or Full   │      │
                         │     RBY Exchange)       │      │
                         └───────────┬─────────────┘      │
                                     │                    │
                                     │                    │
                                     ▼                    │
                         ┌─────────────────────────┐      │
                         │    COMPRESSION PHASE    │◄─────┘
                         │                         │
                         │ ┌─────────────────────┐ │
                         │ │   MEMORY DECAY      │ │
                         │ │ • decay_text()      │ │
                         │ │ • create_visual_    │ │
                         │ │   glyph()          │ │
                         │ │ • decay_to_glyph() │ │
                         │ │ • _extract_tags()  │ │
                         │ │ • _calculate_      │ │
                         │ │   quality()        │ │
                         │ └─────────────────────┘ │
                         └───────────┬─────────────┘
                                     │
                                     │
                                     ▼
               ┌───────────────────────────────────────────────────┐
               │      OUTPUT MEMORY & COMPRESSION SYSTEM          │
               │      (RECURSIVE "EXCRETION" STORAGE)             │
               │                                                  │
               │ • Excretion Repository                           │
               │ • Recursive Compression Engine                   │
               │ • Glyph Encoder/Decoder                          │
               │ • Absularity Monitor                             │
               │ • DNA Memory System                              │
               │   ▹ Compressed Glyph Storage                     │
               │   ▹ Memory Strands & Tagging                     │
               │   ▹ Evolution History Tracking                   │
               │   ▹ Memory Consolidation                         │
               │   ▹ File-based & In-Memory Storage              │
               └───────────────────┬───────────────────────────────┘
                                   │
                                   ▼
               ┌───────────────────────────────────────────────────┐
               │          AEOS CLUSTER FRAMEWORK                   │
               │                                                   │
               │ ┌───────────────────────────────────────────────┐ │
               │ │   🛠️ ClusterConstants                         │ │
               │ │ • RBY_WEIGHTS                                │ │
               │ │ • BATCH_SIZE, GLYPH_SIZE                     │ │
               │ │ • PORT_RANGE, BROADCAST_PORT                 │ │
               │ │ • PRIORITY_WEIGHTS                           │ │
               │ │ • TASK_ALLOCATION                           │ │
               │ │ • POWER_LEVELS                              │ │
               │ │ • P2P settings                              │ │
               │ │ • STORAGE_BALANCING                         │ │
               │ │ • HEALTH_CHECK_INTERVAL                     │ │
               │ │ • Security & framework configs              │ │
               │ │                                             │ │
               │ └───────────────────────────────────────────────┘ │
               │                                                   │
               │ ┌───────────────────────────────────────────────┐ │
               │ │   🛠️ NodeDiscovery Class                      │ │
               │ │ • __init__()                                 │ │
               │ │   ▹ purpose: Initialize node discovery       │ │
               │ │   ▹ called_by: AEOS_Cluster                  │ │
               │ │   ▹ depends_on: threading, socket, psutil    │ │
               │ │   ▹ outputs_to: active_nodes dict            │ │
               │ │ • _check_module_available()                  │ │
               │ │   ▹ purpose: Check if module can be imported │ │
               │ │   ▹ called_by: NodeDiscovery.__init__        │ │
               │ │   ▹ depends_on: Python import system         │ │
               │ │   ▹ outputs_to: p2p_enabled flag            │ │
               │ │ • _create_hardware_profile()                 │ │
               │ │   ▹ purpose: Profile node hardware           │ │
               │ │   ▹ called_by: NodeDiscovery.__init__        │ │
               │ │   ▹ depends_on: psutil, torch                │ │
               │ │   ▹ outputs_to: hardware_profile dict        │ │
               │ │ • _determine_power_level()                   │ │
               │ │   ▹ purpose: Assign node power level (1-7)   │ │
               │ │   ▹ called_by: _create_hardware_profile      │ │
               │ │   ▹ depends_on: POWER_LEVELS constant        │ │
               │ │   ▹ outputs_to: power_level attribute        │ │
               │ │ • _enhance_node_with_hardware_profile()      │ │
               │ │   ▹ purpose: Add hardware details to node    │ │
               │ │   ▹ called_by: NodeDiscovery.__init__        │ │
               │ │   ▹ depends_on: hardware_profile dict        │ │
               │ │   ▹ outputs_to: this_node dict               │ │
               │ │ • _calculate_node_priority()                 │ │
               │ │   ▹ purpose: Calculate node task priority    │ │
               │ │   ▹ called_by: _enhance_node_with_hardware   │ │
               │ │   ▹ depends_on: PRIORITY_WEIGHTS constant    │ │
               │ │   ▹ outputs_to: priority attribute           │ │
               │ │ • _generate_private_key()                    │ │
               │ │   ▹ purpose: Generate RSA key for security   │ │
               │ │   ▹ called_by: NodeDiscovery.__init__        │ │
               │ │   ▹ depends_on: rsa crypto library           │ │
               │ │   ▹ outputs_to: private_key attribute        │ │
               │ │ • _enhance_node_info()                       │ │
               │ │   ▹ purpose: Add P2P data to node profile    │ │
               │ │   ▹ called_by: NodeDiscovery.__init__        │ │
               │ │   ▹ depends_on: uuid, this_node dict         │ │
               │ │   ▹ outputs_to: this_node dict with P2P data │ │
               │ │ • _determine_node_roles()                    │ │
               │ │   ▹ purpose: Assign server/storage/etc roles │ │
               │ │   ▹ called_by: _enhance_node_info            │ │
               │ │   ▹ depends_on: hardware profile, constants  │ │
               │ │   ▹ outputs_to: roles list in this_node      │ │
               │ │ • start_discovery()                          │ │
               │ │   ▹ purpose: Launch discovery threads & P2P  │ │
               │ │   ▹ called_by: AEOS_Cluster                  │ │
               │ │   ▹ depends_on: threading, P2P network       │ │
               │ │   ▹ outputs_to: discovery_thread             │ │
               │ │ • _discovery_loop()                          │ │
               │ │   ▹ purpose: Main node discovery loop        │ │
               │ │   ▹ called_by: start_discovery thread        │ │
               │ │   ▹ depends_on: _network_scan                │ │
               │ │   ▹ outputs_to: known_nodes dict             │ │
               │ │ • _network_scan()                            │ │
               │ │   ▹ purpose: Find nodes on network           │ │
               │ │   ▹ called_by: _discovery_loop               │ │
               │ │   ▹ depends_on: socket, P2P network          │ │
               │ │   ▹ outputs_to: known_nodes via _register    │ │
               │ │ • _register_node()                           │ │
               │ │   ▹ purpose: Register a discovered node      │ │
               │ │   ▹ called_by: _network_scan, _handle_client │ │
               │ │   ▹ depends_on: datetime, node_lock          │ │
               │ │   ▹ outputs_to: known/active nodes dicts     │ │
               │ │ • _health_check_loop()                       │ │
               │ │   ▹ purpose: Monitor node health status      │ │
               │ │   ▹ called_by: start_discovery thread        │ │
               │ │   ▹ depends_on: _check_node_health           │ │
               │ │   ▹ outputs_to: node_health dict             │ │
               │ │ • _check_node_health()                       │ │
               │ │   ▹ purpose: Test connection to remote node  │ │
               │ │   ▹ called_by: _health_check_loop            │ │
               │ │   ▹ depends_on: socket, json                 │ │
               │ │   ▹ outputs_to: node_health status update    │ │
               │ │ • start_server()                             │ │
               │ │   ▹ purpose: Start node server for requests  │ │
               │ │   ▹ called_by: AEOS_Cluster                  │ │
               │ │   ▹ depends_on: threading                    │ │
               │ │   ▹ outputs_to: server_thread                │ │
               │ │ • _server_loop()                             │ │
               │ │   ▹ purpose: Accept incoming connections     │ │
               │ │   ▹ called_by: start_server thread           │ │
               │ │   ▹ depends_on: socket                       │ │
               │ │   ▹ outputs_to: client handler threads       │ │
               │ │ • _handle_client()                           │ │
               │ │   ▹ purpose: Process client connection       │ │
               │ │   ▹ called_by: _server_loop                  │ │
               │ │   ▹ depends_on: _process_request             │ │
               │ │   ▹ outputs_to: client socket response       │ │
               │ │ • _process_request()                         │ │
               │ │   ▹ purpose: Handle different request types  │ │
               │ │   ▹ called_by: _handle_client                │ │
               │ │   ▹ depends_on: request handlers             │ │
               │ │   ▹ outputs_to: response dict                │ │
               │ │ • _handle_task_request()                     │ │
               │ │   ▹ purpose: Process task execution request  │ │
               │ │   ▹ called_by: _process_request              │ │
               │ │   ▹ depends_on: task distribution system     │ │
               │ │   ▹ outputs_to: task_response dict           │ │
               │ │ • _handle_status_request()                   │ │
               │ │   ▹ purpose: Provide node cluster status     │ │
               │ │   ▹ called_by: _process_request              │ │
               │ │   ▹ depends_on: node_lock, active_nodes      │ │
               │ │   ▹ outputs_to: status_response dict         │ │
               │ │ • _handle_code_analysis_request()            │ │
               │ │   ▹ purpose: Process code analysis request   │ │
               │ │   ▹ called_by: _process_request              │ │
               │ │   ▹ depends_on: analyze_node_codebase        │ │
               │ │   ▹ outputs_to: code_analysis_response dict  │ │
               │ │ • analyze_node_codebase()                    │ │
               │ │   ▹ purpose: Analyze local/remote code       │ │
               │ │   ▹ called_by: _handle_code_analysis_request │ │
               │ │   ▹ depends_on: component_analyzer           │ │
               │ │   ▹ outputs_to: code_analysis_data dict      │ │
               │ │ • _request_code_analysis()                   │ │
               │ │   ▹ purpose: Get code analysis from node     │ │
               │ │   ▹ called_by: analyze_node_codebase         │ │
               │ │   ▹ depends_on: socket, json                 │ │
               │ │   ▹ outputs_to: analysis data dict           │ │
               │ │ • get_cluster_component_balance()            │ │
               │ │   ▹ purpose: Calculate cluster RBY balance   │ │
               │ │   ▹ called_by: AEOS_Cluster                  │ │
               │ │   ▹ depends_on: code_analysis_data           │ │
               │ │   ▹ outputs_to: balance dict                 │ │
               │ │ • get_cluster_code_complexity()              │ │
               │ │   ▹ purpose: Analyze overall code metrics    │ │
               │ │   ▹ called_by: AEOS_Cluster                  │ │
               │ │   ▹ depends_on: code_analysis_data           │ │
               │ │   ▹ outputs_to: metrics dict                 │ │
               │ │ • get_active_nodes()                         │ │
               │ │   ▹ purpose: Retrieve list of active nodes   │ │
               │ │   ▹ called_by: AEOS_Cluster                  │ │
               │ │   ▹ depends_on: node_lock, active_nodes      │ │
               │ │   ▹ outputs_to: nodes list                   │ │
               │ │ • get_node_priorities()                      │ │
               │ │   ▹ purpose: Get nodes sorted by priority    │ │
               │ │   ▹ called_by: AEOS_Cluster                  │ │
               │ │   ▹ depends_on: active_nodes dict            │ │
               │ │   ▹ outputs_to: sorted nodes list            │ │
               │ │ • get_node_count()                           │ │
               │ │   ▹ purpose: Count active and known nodes    │ │
               │ │   ▹ called_by: AEOS_Cluster                  │ │
               │ │   ▹ depends_on: node_lock, node dicts        │ │
               │ │   ▹ outputs_to: count dict                   │ │
               │ └───────────────────────────────────────────────┘ │
               │                                                   │
               │ ┌───────────────────────────────────────────────┐ │
               │ │   🛠️ TaskDistributor Class                    │ │
               │ │ • __init__()                                 │ │
               │ │   ▹ purpose: Initialize task distributor     │ │
               │ │   ▹ called_by: main()                        │ │
               │ │   ▹ depends_on: queue, threading, logger     │ │
               │ │   ▹ outputs_to: task_distributor object      │ │
               │ │ • start()                                    │ │
               │ │   ▹ purpose: Start task distribution thread  │ │
               │ │   ▹ called_by: main()                        │ │
               │ │   ▹ depends_on: threading                    │ │
               │ │   ▹ outputs_to: task_thread                  │ │
               │ │ • stop()                                     │ │
               │ │   ▹ purpose: Stop task distribution thread   │ │
               │ │   ▹ called_by: main() on shutdown            │ │
               │ │   ▹ depends_on: task_thread                  │ │
               │ │   ▹ outputs_to: stop_thread flag             │ │
               │ │ • _task_loop()                              │ │
               │ │   ▹ purpose: Background task distribution    │ │
               │ │   ▹ called_by: start() thread                │ │
               │ │   ▹ depends_on: task_queue                   │ │
               │ │   ▹ outputs_to: _distribute_task calls       │ │
               │ │ • _distribute_task()                        │ │
               │ │   ▹ purpose: Assign task to appropriate node │ │
               │ │   ▹ called_by: _task_loop                    │ │
               │ │   ▹ depends_on: _find_suitable_nodes         │ │
               │ │   ▹ outputs_to: _send_task_to_node           │ │
               │ │ • _find_suitable_nodes()                    │ │
               │ │   ▹ purpose: Match task to capable nodes     │ │
               │ │   ▹ called_by: _distribute_task              │ │
               │ │   ▹ depends_on: ClusterConstants, node_discovery│ │
               │ │   ▹ outputs_to: suitable nodes list          │ │
               │ │ • _send_task_to_node()                      │ │
               │ │   ▹ purpose: Send task to selected node      │ │
               │ │   ▹ called_by: _distribute_task              │ │
               │ │   ▹ depends_on: socket, json                 │ │
               │ │   ▹ outputs_to: node socket                  │ │
               │ │ • submit_task()                             │ │
               │ │   ▹ purpose: Add task to distribution queue  │ │
               │ │   ▹ called_by: various components            │ │
               │ │   ▹ depends_on: hashlib, queue               │ │
               │ │   ▹ outputs_to: task_queue                   │ │
               │ │ • get_task_status()                         │ │
               │ │   ▹ purpose: Check state of submitted task   │ │
               │ │   ▹ called_by: task requesters               │ │
               │ │   ▹ depends_on: task_lock                    │ │
               │ │   ▹ outputs_to: status dict                  │ │
               │ │ • register_result()                         │ │
               │ │   ▹ purpose: Store completed task result     │ │
               │ │   ▹ called_by: task handlers                 │ │
               │ │   ▹ depends_on: task_lock                    │ │
               │ │   ▹ outputs_to: completed_tasks dict         │ │
               │ │ • register_failure()                        │ │
               │ │   ▹ purpose: Record task failure             │ │
               │ │   ▹ called_by: task handlers                 │ │
               │ │   ▹ depends_on: task_lock                    │ │
               │ │   ▹ outputs_to: failed_tasks dict            │ │
               │ └───────────────────────────────────────────────┘ │
               │                                                   │
               │ ┌───────────────────────────────────────────────┐ │
               │ │   💎 DistributedIndex Class                   │ │
               │ │ • __init__()                                 │ │
               │ │   ▹ purpose: Initialize FAISS vector index    │ │
               │ │   ▹ called_by: main()                        │ │
               │ │   ▹ depends_on: faiss, task_distributor      │ │
               │ │   ▹ outputs_to: local_index                  │ │
               │ │ • add()                                      │ │
               │ │   ▹ purpose: Add vectors to distributed index │ │
               │ │   ▹ called_by: vector embedding systems      │ │
               │ │   ▹ depends_on: torch, numpy, index_lock     │ │
               │ │   ▹ outputs_to: local_index, remote nodes    │ │
               │ │ • search()                                   │ │
               │ │   ▹ purpose: Search vectors by similarity     │ │
               │ │   ▹ called_by: retrieval systems             │ │
               │ │   ▹ depends_on: torch, numpy, index_lock     │ │
               │ │   ▹ outputs_to: distances and indices arrays │ │
               │ │ • _distribute_vectors()                      │ │
               │ │   ▹ purpose: Sync vectors to other nodes     │ │
               │ │   ▹ called_by: add()                         │ │
               │ │   ▹ depends_on: zlib, pickle, task_distributor│ │
               │ │   ▹ outputs_to: index_update tasks           │ │
               │ └───────────────────────────────────────────────┘ │
               │                                                   │
               │ ┌───────────────────────────────────────────────┐ │
               │ │   🛠️ DistributedStorage Class                 │ │
               │ │ • __init__()                                 │ │
               │ │   ▹ purpose: Initialize distributed storage   │ │
               │ │   ▹ called_by: main()                        │ │
               │ │   ▹ depends_on: node_discovery, threading     │ │
               │ │   ▹ outputs_to: db_conn, storage system       │ │
               │ │ • _init_db()                                 │ │
               │ │   ▹ purpose: Set up SQLite storage database   │ │
               │ │   ▹ called_by: __init__                       │ │
               │ │   ▹ depends_on: sqlite3                       │ │
               │ │   ▹ outputs_to: db_conn object                │ │
               │ │ • store_item()                               │ │
               │ │   ▹ purpose: Store data with redundancy       │ │
               │ │   ▹ called_by: various data storage systems   │ │
               │ │   ▹ depends_on: p2p_network, zlib            │ │
               │ │   ▹ outputs_to: db, file storage, remote nodes│ │
               │ │ • _store_local()                             │ │
               │ │   ▹ purpose: Save data to local filesystem    │ │
               │ │   ▹ called_by: store_item                     │ │
               │ │   ▹ depends_on: os, filesystem                │ │
               │ │   ▹ outputs_to: compressed binary file        │ │
               │ │ • _distribute_item()                         │ │
               │ │   ▹ purpose: Send item to other nodes         │ │
               │ │   ▹ called_by: store_item                     │ │
               │ │   ▹ depends_on: base64, task_distributor     │ │
               │ │   ▹ outputs_to: store_item tasks             │ │
               │ │ • _get_suitable_storage_nodes()              │ │
               │ │   ▹ purpose: Find nodes with available space  │ │
               │ │   ▹ called_by: store_item                     │ │
               │ │   ▹ depends_on: node_discovery               │ │
               │ │   ▹ outputs_to: suitable nodes list           │ │
               │ │ • retrieve_item()                            │ │
               │ │   ▹ purpose: Get item from storage system     │ │
               │ │   ▹ called_by: data retrieval components      │ │
               │ │   ▹ depends_on: storage_lock, zlib           │ │
               │ │   ▹ outputs_to: decompressed data             │ │
               │ │ • _fetch_from_remote()                       │ │
               │ │   ▹ purpose: Get item from another node       │ │
               │ │   ▹ called_by: retrieve_item                  │ │
               │ │   ▹ depends_on: storage_lock, _fetch_from_node│ │
               │ │   ▹ outputs_to: item data or None             │ │
               │ │ • _fetch_from_node()                         │ │
               │ │   ▹ purpose: Request item from specific node  │ │
               │ │   ▹ called_by: _fetch_from_remote             │ │
               │ │   ▹ depends_on: socket, json                  │ │
               │ │   ▹ outputs_to: decompressed data or None     │ │
               │ │ • _find_node_by_id()                         │ │
               │ │   ▹ purpose: Locate node from its ID          │ │
               │ │   ▹ called_by: _fetch_from_remote             │ │
               │ │   ▹ depends_on: node_discovery               │ │
               │ │   ▹ outputs_to: node dict or None             │ │
               │ │ • _node_id()                                 │ │
               │ │   ▹ purpose: Generate consistent node ID      │ │
               │ │   ▹ called_by: various storage functions      │ │
               │ │   ▹ depends_on: hashlib                       │ │
               │ │   ▹ outputs_to: md5 hash string               │ │
               │ │ • _this_node_id()                            │ │
               │ │   ▹ purpose: Get current node's unique ID     │ │
               │ │   ▹ called_by: various storage functions      │ │
               │ │   ▹ depends_on: node_discovery, _node_id      │ │
               │ │   ▹ outputs_to: node ID string                │ │
               │ └───────────────────────────────────────────────┘ │
               │                                                   │
               │ ┌───────────────────────────────────────────────┐ │
               │ │   🛠️ AEOS Cluster CLI & Main Functions        │ │
               │ │ • parse_args()                               │ │
               │ │   ▹ purpose: Process command line arguments   │ │
               │ │   ▹ called_by: main()                        │ │
               │ │   ▹ depends_on: argparse                      │ │
               │ │   ▹ outputs_to: args namespace object         │ │
               │ │ • main()                                     │ │
               │ │   ▹ purpose: Initialize & run cluster node    │ │
               │ │   ▹ called_by: script execution               │ │
               │ │   ▹ depends_on: all cluster components        │ │
               │ │   ▹ outputs_to: running cluster node          │ │
               │ └───────────────────────────────────────────────┘ │
               │                                                   │
               │ ┌───────────────────────────────────────────────┐ │
               │ │   🔄 P2P Version Management Integration       │ │
               │ │ • get_p2p_network()                          │ │
               │ │   ▹ purpose: Access P2P networking services   │ │
               │ │   ▹ called_by: ComponentEvolution.__init__    │ │
               │ │   ▹ depends_on: aeos_p2p_network              │ │
               │ │   ▹ outputs_to: p2p_version_manager           │ │
               │ │ • backup_file()                              │ │
               │ │   ▹ purpose: Create P2P backup of file        │ │
               │ │   ▹ called_by: _apply_improvement             │ │
               │ │   ▹ depends_on: file_version_manager          │ │
               │ │   ▹ outputs_to: distributed backup            │ │
               │ │ • restore_file()                             │ │
               │ │   ▹ purpose: Restore file from P2P backup     │ │
               │ │   ▹ called_by: _apply_improvement (failure)   │ │
               │ │   ▹ depends_on: file_version_manager          │ │
               │ │   ▹ outputs_to: restored original file        │ │
               │ └───────────────────────────────────────────────┘ │
               │                                                   │
               └───────────────────┬───────────────────────────────┘
                                   │
                                   ▼
               ┌───────────────────────────────────────────────────┐
               │             DEPLOYMENT & PUBLICATION MANAGER      │
               │                                                   │
               │ • Auto-Deploy Engine                             │
               │ • Publication & Integration Handler              │
               │ • Safety & Compliance Checker                    │
               │ • User Guidance Prompter                         │
               │ • Third-Party Integration Modules                │
               │ • Excretion Points Management                    │
               │   ▹ Function/Method Identification               │
               │   ▹ Integration Graph Building                    │
               └───────────────────┬───────────────────────────────┘
                                   │
                                   ▼
               ┌───────────────────────────────────────────────────┐
               │   AUTOMATED DOCUMENTATION & USER GUIDANCE         │
               │                                                   │
               │ • Text Documentation Generator                    │
               │ • Interactive Tutorial Scripts                    │
               │ • Video/Audio Guide Creator                      │
               │ • Third-Party Process Explainer                  │
               │ • Packaging and Presentation                     │
               │ • Logging System                                 │
               │   ▹ File & Stream Handlers                       │
               │   ▹ Error Tracking & Analysis                    │
               └───────────────────┬───────────────────────────────┘
                                   │
                                   ▼
               ┌───────────────────────────────────────────────────┐
               │                      AE                           │
               │             ABSOLUTE EXISTENCE                    │
               │                                                   │
               │  • Source repository (user's PC/files)            │
               │  • Immutable, inert reference layer               │
               │  • Houses compressed glyphs from all cycles       │
               │  • Read-only except for C-AE deposits             │
               │                                                   │
               │ ┌─────────────────────────────────────────┐       │
               │ │      ABSOLUTE EXISTENCE CLASS           │       │
               │ │ • load_glyphs()                         │       │
               │ │ • get_glyph_by_id()                     │       │
               │ │ • get_glyphs_by_tag()                   │       │
               │ │ • store_glyph()                         │       │
               │ │ • get_dna_glyph_by_tag()                │       │
               │ │ • select_seed_for_expansion()           │       │
               │ │ • get_storage_percentage()              │       │
               │ │ • get_glyph_count()                     │       │
               │ └─────────────────────────────────────────┘       │
               └───────────────────┬───────────────────────────────┘
                                   │
                                   │
                                   ▼
                        NEW EXPANSION BEGINS
                       FROM PREVIOUS GLYPH SEED
```

## Key Concepts in the RBY Singularity Framework

### Core Components
- **RBY Singularity**: Launch script that initiates the entire system with seed values (UF+IO = RBY)
- **Launch & Initialization Orchestrator**: Bootstraps the entire AI organism with proper sequence and configuration
- **User Interaction Interface**: Handles all user communications via terminal and graphical interfaces
- **Natural Language & Generative Engine**: Core "brain" that processes inputs and generates outputs
- **LLM Training Pipeline**: Self-learning mechanism that continuously improves the AI models
- **C-AE (Crystallized AE)**: The dynamic expansion environment, the only "moving" component
- **Multimodal Media Generation**: Creates images, videos, audio, and 3D content to complement text
- **IC-AE**: Infected C-AE sandboxes created when scripts enter C-AE
- **Absularity**: Maximum expansion limit (90% storage or complete RBY exchange)
- **Output Memory & Compression**: Long-term memory using recursive compression and glyph encoding
- **Deployment & Publication**: Manages auto-publishing and deployment of generated outputs
- **Documentation & Guidance**: Creates user guides and tutorials for the generated solutions
- **AE (Absolute Existence)**: Immutable source layer (user's PC/files) that provides foundation
- **Constants Configuration**: System-wide constants for paths, thresholds, and settings
- **DNA Memory System**: Long-term storage system for compressed knowledge glyphs
- **RBY Component System**: Classifies and manages code components according to RBY architecture
- **Integration Levels**: Defines how tightly components are coupled (LOOSE, MEDIUM, TIGHT)
- **Component Analyzer**: Discovers, analyzes, and maps code components and their relationships
- **Memory Decay**: System for compressing knowledge through progressive abstraction into glyphs
- **CrystalizedAE Class**: Manages the expansion/compression cycle and evolution of the system
- **AEOS Cluster Framework**: Manages distributed computing across multiple nodes with varying capabilities
- **Node Discovery**: Discovers and manages cluster nodes with P2P networking capabilities
- **Hardware Profiling**: Creates detailed hardware profiles to optimize task allocation
- **TaskDistributor**: Distributes computational tasks across cluster nodes based on hardware capabilities
- **DistributedIndex**: Manages vector embeddings across nodes with FAISS for semantic search
- **DistributedStorage**: Provides redundant P2P-enhanced storage across cluster nodes

### Process Cycle
1. **RBY Singularity** initiates the launch orchestrator to bootstrap the system
2. **Launch Orchestrator** spawns all system components in proper sequence
3. **User Interface** accepts input and presents outputs back to users
4. **Generative Engine** processes requests and creates responses (text, code, etc.)
5. **Expansion Phase** creates C-AE shell where multimodal outputs are generated
6. **IC-AE Creation** - Scripts enter C-AE and become infected with singularity
7. **Recursive Infection** - Each IC-AE creates nested sandboxes (IIC-AE)
8. **Absularity** reached when expansion hits maximum threshold
9. **Compression Phase** begins, using the Output Memory system
10. **Deployment Manager** handles publication of outputs to target environments
11. **Documentation Generator** creates guides for using the generated outputs
12. **Knowledge Deposit into AE** - Compressed knowledge enters immutable storage
13. **New Expansion** begins using previous glyph seed

### Intelligence Trifecta Structure
- **R (Red)**: Perception weight - environmental awareness, input processing
- **B (Blue)**: Cognition weight - logical processing, understanding
- **Y (Yellow)**: Execution weight - action manifestation, output generation

### Self-Evolution Mechanisms
- **Continuous Monitoring & Self-Modification**: Enables the AI to adapt and improve itself
- **Distributed Compute Network**: Provides scalable resources for training and heavy processing
- **Training Pipeline**: Allows the AI to learn from past interactions and outputs
- **Memory Compression**: Prevents bloat through progressive abstraction of knowledge
- **Component Classification**: Analyzes code to determine RED/BLUE/YELLOW roles
- **Integration Mapping**: Discovers how components interact and integrate
- **DNA Memory**: Long-term storage with evolution tracking and memory consolidation
- **Resource Monitoring**: Tracks CPU/RAM usage and triggers compression when needed
- **Memory Decay Process**: Progressive text abstraction through multiple decay levels
- **Evolution Tracking**: Records evolutionary events and changes in the system
- **Glyph Visual Representation**: Creates visual patterns representing compressed knowledge
- **Node Discovery & P2P Networking**: Discovers and connects cluster nodes dynamically
- **Hardware-Optimized Task Allocation**: Assigns tasks based on node capabilities and power levels
- **RBY-Based Component Classification**: Determines red/blue/yellow roles in code architecture
- **ComponentEvolution**: Self-evolution and optimization system for code components
- **P2P Version Management**: Distributed backup and versioning system for evolved code

### AEOS Cluster Features
- **Node Registration**: Tracks discovered nodes and their capabilities
- **Health Monitoring**: Continuous health checks of all connected nodes
- **Server Management**: Handles incoming connections and requests
- **Code Analysis**: Analyzes codebase across the cluster for RBY balance
- **Task Processing**: Distributes and processes tasks across nodes
- **Cluster Metrics**: Tracks complexity, component balance, and integration
- **Priority-Based Task Allocation**: Assigns tasks based on node capability
- **P2P Communication**: Enables node-to-node direct communication
- **Dynamic Network Discovery**: Automatically finds and integrates new nodes
- **Fault Tolerance**: Handles node failures and maintains cluster integrity
- **Task Distribution**: Intelligently routes tasks to best-suited nodes based on hardware
- **Distributed Vector Index**: Maintains semantic search vectors across cluster nodes
- **Redundant Storage**: Stores data with configurable redundancy across multiple nodes
- **P2P Enhanced Data Transfer**: Uses peer-to-peer protocols for efficient data sharing
- **Hardware-Aware Computing**: Matches computational tasks to node capabilities
- **CLI Configuration**: Command-line interface for cluster node setup and management

### Component Evolution System
- **Evolution Console**: Interactive interface for managing code evolution
- **Function Registry**: Tracks available functions for optimization
- **Sandbox Testing**: Tests evolved code in isolated environments
- **Code Mutation**: Generates potential improvements to functions
- **Evolution History**: Tracks successful and failed optimization attempts
- **Singularity Integration**: Enhanced mutations via Singularity framework 
- **P2P Version Management**: Distributed backup of evolved code files
- **Codebase Scanning**: Automatically discovers optimizable components
- **AST-based Optimization**: Uses abstract syntax trees to analyze and modify code
- **RBY-Specific Optimization**: Different strategies for Red, Blue, and Yellow components
- **Performance Benchmarking**: Measures function performance before and after optimization
- **Test Case Generation**: Automatically creates test cases based on function signatures
- **Loop Optimization**: Identifies and improves loop efficiency in code
- **Operation Combination**: Consolidates redundant operations for better performance
- **Function Caching**: Adds memoization for computationally expensive functions
- **Backup and Versioning**: Creates safe backups before applying optimizations
- **Status Reporting**: Provides detailed evolution metrics and performance statistics
- **Background Evolution**: Autonomous optimization thread for continuous improvement
