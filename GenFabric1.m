function GenFabric1(CELL,out_mat,VF, h, out_mat_dir, ply_out_dir);
    [mat_out_dir, name, ext] = fileparts(out_mat);
    filename_base = name; 
    % Construct the full path to the .mat file safely
    if strcmpi(ext, '.mat')
        mat_path_full = out_mat;
    else
        mat_path_full = fullfile(mat_out_dir, [name '.mat']);
    end

    % --- 2. Load Data ---
    if ~exist(mat_path_full, 'file')
        error('GenFabric2D: Input file not found: %s', mat_path_full);
    end
    %load(mat_path_full, 'CELL');
    fprintf('DEBUG: Calling Thirdphasegeneration with VF = %.4f, h = %.4f\n', VF, h);
    Equation = 4;
    Parameters = [VF, h]; 
    [CELL, SURF_BINDING, RVE] = Thirdphasegeneration(CELL,Equation,Parameters); 
    
    % Extract inclusion surfaces and flatten to row vector
    SURF_INCLUSIONS = [CELL.INCL.CONT];
    SURF_INCLUSIONS = SURF_INCLUSIONS(:)';
    
    % Wrap in a cell array to keep them distinct
    TOTAL_SURF = {SURF_INCLUSIONS, SURF_BINDING};

    CELL.RVE = RVE;
    %save("-mat7-binary", mat_path_full, "CELL");
    %disp(["saved mat file"]);
    save_and_plot_phi_hard(CELL, TOTAL_SURF, filename_base, mat_out_dir,ply_out_dir);
    write_void_mask_tiff(CELL,RVE,filename_base, ply_out_dir); 

% --- Nested Functions ---

function save_and_plot_phi_hard(CELL, TOTAL_SURF, filename, mat_out_dir, ply_out_dir)
    if nargin < 4 || isempty(mat_out_dir), mat_out_dir = pwd; end
    if nargin < 5 || isempty(ply_out_dir), ply_out_dir = mat_out_dir; end
    
    if ~exist(mat_out_dir, 'dir'), mkdir(mat_out_dir); end
    if ~exist(ply_out_dir, 'dir'), mkdir(ply_out_dir); end


    
    % Color Logic
    COL_INCL_BASE = col_cell(CELL, 'incl', 42);
    % Reshape to 3 rows (RGB) x (7 patches * N inclusions)
    COL_INCL = reshape(repmat(COL_INCL_BASE', [7 1]), [3, CELL.NUM_INCL*7]);
    my_blue = [0.0; 0.0; 0.5]; % Column vector

    % --- Plotting ---
    figure('visible', 'off'); 
    hold on;
    axis(CELL.LIM(:)'); 
    axis equal; 
    view(3);
    S_INC = TOTAL_SURF{1};
    S_BIN = TOTAL_SURF{2};

    for k = 1:numel(S_INC)
        plt_surf(S_INC(k), 'FaceColor', COL_INCL(:,k)', 'EdgeColor', 'none');
    end

    for k = 1:numel(S_BIN)
        plt_surf(S_BIN(k), 'FaceColor', my_blue', 'FaceAlpha', 0.4, 'EdgeColor', 'none');
    end
    camlight; lighting gouraud;

    % --- Prepare for Export ---
    % Flatten both into a single row of structures
    S_INC_ROW = S_INC(:)';
    S_BIN_ROW = S_BIN(:)';
    FINAL_SURF_LIST = [S_INC_ROW, S_BIN_ROW];
    
    % FIX: Ensure color matrix matches the number of surface patches
    COL_BINDING_REP = repmat(my_blue, 1, numel(S_BIN));
    COL_TOTAL = [double(COL_INCL), double(COL_BINDING_REP)];

    % --- Save MAT ---
    matfile = fullfile(mat_out_dir, [filename '.mat']);
    if exist('OCTAVE_VERSION','builtin')
        save('-mat7-binary', matfile, 'CELL', 'FINAL_SURF_LIST');
    else
        save(matfile, 'CELL', 'FINAL_SURF_LIST', '-v7.3');
    end
    disp(['Saved MAT to: ', matfile]);

    % --- Save PLY ---
    plybase = fullfile(ply_out_dir, [filename '_phi_hard']);
    ply_surf_ox(plybase, FINAL_SURF_LIST, COL_TOTAL);
    disp(['Saved PLY to: ', plybase, '.ply']);
end


function write_void_mask_tiff(CELL ,RVE ,filename, ply_out_dir)


    if ~exist(ply_out_dir, 'dir'), mkdir(ply_out_dir); end

    void_u8 = uint8(RVE < 0) * 255;

    % 2) Write each z-slice as a page in a TIFF

    tifname  = fullfile(ply_out_dir, [filename '_void_mask.tif']);
    if exist(tifname, 'file')
        delete(tifname);
    end


    for k = 1:size(void_u8, 3)
        slice = void_u8(:,:,k);
        if ndims(slice) == 3
            slice = slice(:,:,1);
        end
        if k == 1
            imwrite(slice, tifname, 'tif', 'Compression','none');
        else
            imwrite(slice, tifname, 'tif', ...
                    'WriteMode','append', 'Compression','none');
        end
    end
    clear void_u8;
    fprintf('Saved binary void mask stack to: %s\n', tifname);
end
end
