function [CELL] = EXAMPLE_03_HELLO(DIM, seed, X, out_path,out_ply_base)

   try
    rand("seed", seed);
    randn("seed", seed);
  catch
    rand("state", seed);
    randn("state", seed);
  end                             % to generate same RVE 

  run("INCL-h.m");
  run("CELL-h.m");
  run("PACK-h.m");

  % ---- decode X (input vector) ----
  L = numel(X);
  B = 5;             % Bins of inclusion size
  VF   = X(1);
  rmin = X(4);
  rmax = X(5);
  xs = X(5+B); ys = X(5+B+1); zs = X(5+B+2);             % ELONGATION RATIOS
  dvec = sort(X(5+B+3 : 5+B+2+DIM),'ascend');   % 3  NND values
  disp(VF);

  radius_block = X(6 : 6 + (B-2));
  internal_edges = radius_block  # length B-1 (last slot is kappa)
  edges = [rmin; internal_edges(:); rmax];
  e_lower = edges(1:end-1);
  e_upper = edges(2:end);
  CURV.SIZ = fliplr([e_lower(:)' ; e_upper(:)']);
  CURV.VOL = repmat(VF / B, B, 1);
  %CURV.VOL = round(CURV.VOL * 100000) / 100000;
  CURV.NND = repmat(dvec(:), 1, B);
  fprintf('CURV.NND size: [%d x %d]\n', size(CURV.NND,1), size(CURV.NND,2));
  disp(CURV.NND);
  % //////////////////////////////////////////////////////////////////
  % /// DEBUGGING: INSPECT CURV BEFORE PROCEEDING                  ///
  % //////////////////////////////////////////////////////////////////
  disp("============================================================");
  disp("DEBUG: INSPECTING CURV STRUCTURE");
  
  disp("1. CURV.SIZ (Row 1=Min, Row 2=Max):");
  disp(CURV.SIZ);
  disp(VF);

  
  disp("2. CURV.VOL (Volume Fractions):");
  disp(CURV.VOL);
  
  % Check if dimensions match (Number of columns in SIZ == Number of rows in VOL)
  num_bins = size(CURV.SIZ, 2);
  num_vols = size(CURV.VOL, 1);
  
  printf("3. DIMENSION CHECK:\n");
  printf("   Bins from SIZ: %d\n", num_bins);
  printf("   Bins from VOL: %d\n", num_vols);
  
  if (num_bins != num_vols)
      disp("   !!! WARNING: MISMATCH DETECTED !!!");
  else
      disp("   STATUS: OK (Dimensions match)");
  end
  
  disp("============================================================");
  % ---- inclusion shape IOP ----
  IOP.DIM     = DIM;
  IOP.MET     = "RNOI_HER";
  IOP.PRE     = 15;
  IOP.RAT     = [xs ys zs](1:DIM);
  IOP.NOI_PRE = [4,4];
  IOP.NOI_INT = [0.0,0.0];

  NI   = 10;
  INCL = gen_incl(IOP, NI);




  % ---- CELL ----
  COP = nul_cop(DIM);
  COP.LIM = [0 1 0 1 {0 1}{DIM==3}]';
  if (DIM==2) COP.PRE = [200]; else COP.PRE = [200]; end
  COP.BOR = "PER";
  %COP.BOR = "BOX";                                                          
  COP.NAM = "From X+seed";
  COP.NUM_DNK = DIM;
  RUDE_DISP_                        ( "Initializing CELL ..." );                                      % 
  CELL = ini_cell(COP);

  % ---- POP ----
  POP.NAM = "Inclusion population";
  POP.TAR_MOD = "VOL";
  POP.TAR_VOL = VF;
  POP.TAR_TOT = false;

  POP.POS_MOD = "NND";
  POP.GAP = 0.0;
  POP.NND = dvec(:);

  POP.SIZ = [rmin; rmax];
  POP.ANG_RND = 1.0;
  if (DIM==3) POP.ANG_ROT = [1 0 0]'; else POP.ANG_ROT = pi/4; end
  POP.DNK_MOD = "BDS";

  POP = pop_curv(CURV, POP);

  % ---- pack ----
  RUDE_DISP_                        ( "Packing inclusions ..." );                                     % 
  tic
  CELL = rsa_pack(CELL, INCL, POP);
  toc


  if                                ( DIM == 2 )
    plt_cell                        ( CELL ); drawnow;                                                % OCTAVE PLOT - 3D OK BUT SLOW
    svg_cell                        ( out_ply_base , CELL );                                      % SVG - 2D ONLY
  elseif                            ( DIM == 3 )
    ply_cell                        ( out_ply_base , CELL );                                      % PLY - 3D ONLY
  end
  if (nargin >= 4 && !isempty(out_path))
    save("-mat7-binary", out_path, "CELL");
  end
end
