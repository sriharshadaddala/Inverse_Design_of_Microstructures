#!/usr/bin/env bash

# 1. Start the container if it's not running
docker start RVE_1900-2000

# 2. External Bash Loop (The "Mac-style" Isolation)
# This forces a clean Octave process for every single i
for i in {2..11}
do
    echo "----------------------------------------------------"
    echo ">>> STARTING RVE INDEX: $i <<<"
    echo "----------------------------------------------------"

    # docker exec runs Octave for ONLY ONE index and then quits
    docker exec -i RVE_1900-2000 bash -ic "
        cd /home/code/RUDE/lab/CELL/1.0 && \
        octave --no-gui --silent --eval \"
            % --- STABILITY HEADERS ---
            crash_dumps_octave_core(0);
            sigterm_dumps_octave_core(0);
            sighup_dumps_octave_core(0);
            graphics_toolkit('gnuplot');
            set(0, 'defaultfigurevisible', 'off');
            addpath('lab/CODE');
            pkg load image;

            % --- LOAD DATA FOR THIS SPECIFIC INDEX ---
            X_unique = dlmread('lab/CODE/OUTPUT/X_unique.csv', ',');
            seeds    = dlmread('lab/CODE/OUTPUT/rve_seeds.csv', ',');
            
            DIM = 3;
            i = $i; % Injected from Bash loop
            k = 1;
            
            X = X_unique(i,:);
            seed = int64(seeds(i,k));

            out_mat_dir = 'lab/CODE/OUTPUT/MAT_FILES';
            out_ply_dir = 'lab/CODE/OUTPUT/PLY_FILES';
            [~,~] = mkdir(out_mat_dir);
            [~,~] = mkdir(out_ply_dir);

            out_mat = sprintf('%s/CELL_i%05d_k%02d_seed%d.mat', out_mat_dir, i, k, seed);
            out_ply = sprintf('%s/CELL_i%05d_k%02d_seed%d',     out_ply_dir, i, k, seed);

            % --- CORE GENERATION ---
            try
                printf('Generating RVE %d with seed %d...\\n', i, seed);
                printf('DEBUG: The variable X has %d elements.\n', numel(X));
                disp('DEBUG: Contents of X:');
                CELL = EXAMPLE_03_HELLO(DIM, seed, X, out_mat, out_ply);
                
                GenFabric1(CELL, out_mat, X(2), X(3), out_mat_dir, out_ply_dir);
                
                % --- TIFF VOID MASK CLEANUP ---
                out_tif = sprintf('%s/CELL_i%05d_k%02d_seed%d_void_mask.tif', out_ply_dir, i, k, seed);
                if exist(out_tif, 'file')
                    info = imfinfo(out_tif);
                    num_slices = numel(info);
                    tmp_tif = [out_tif, '.tmp'];
                    
                    for z = 1:num_slices
                        img = imread(out_tif, z);
                        if ndims(img) == 3, img = img(:,:,1); end
                        
                        if z == 1
                            imwrite(img, tmp_tif, 'tif', 'Compression', 'none');
                        else
                            imwrite(img, tmp_tif, 'tif', 'WriteMode', 'append', 'Compression', 'none');
                        end
                    end
                    delete(out_tif);
                    movefile(tmp_tif, out_tif);
                    fprintf('Cleaned TIFF: %s (%d slices)\\n', out_tif, num_slices);
                end
                
                printf('SUCCESS: Saved i=%d k=%d seed=%d\\n', i, k, seed);
                
            catch ME
                fprintf('FATAL ERROR at index %d: %s\\n', i, ME.message);
                % We don't exit the whole bash script, just this index
            end
            
            % Force clear and exit Octave to flush RAM
            close all;
            clear -all;
        \"
    "
    
    # Optional: Short 2-second rest for the CPU/Disk to breathe between containers
    sleep 2
done