function [CELL, SURF, RVE] = Thirdphasegeneration(CELL,Equation,Parameters)
%%
% ========================================================
% Add a Thid phase to and CELL with Inclusions + Matrix 
% the output CELL will contain : Inclusions, Matrix, Void
% ========================================================
% How to call the function : [CELL] = genThirdPhase(CELL,Equation,Parameters)
% 
% |IN :
% |
% |CELL      = Input CELL
% |Equation  = ID of the equation/methods
% | 1 : Void(x) = a-LS1*LS2^b => the thickness of the coating is imposed (a)
% | 2 : Void(x) = -min(LS1-c,LS1+LS2*d-e) => The coating thickness & the
% bridging length are imposed, c: coating term, e: bridge length
% | 3 : Void(x) = F(f)-LS1*LS2^b the volume fraction of the third phase is
% controlled (f)
% | 4 : Void(x) = -min(LS1-G(g,h),LS1+LS2*1-H(g,h)) Both, volume fraction
% of third phase (g) & the fraction of the bridges in this phase are
% controlled (h)
% |
% |Parameters   = it depends on the equation, default values exist
%
% |DEFAULT PARAMETERS (use Parameters to change) :
% |
% | a = 0.002;      EQ = 1, Coating term
% | b = 1;          EQ = 1,3, LS2 exponent
% | c = 0.005;      EQ = 2, Coating term
% | d = 1;          EQ = 2, LS2 factor 
% | e = 0.045;      EQ = 2, Bridge lenght
% | f = 0.2;        EQ = 3, Fabric fraction
% | g = 0.2;        EQ = 4, Fabric fraction
% | h = 0.5;        EQ = 4, Bridge ration
% ========================================================

%%
% INITIALIZATION
% =========================================================================
% % LS=[];
% % for i =1:length(CELL.INCL)
% %     LS(:,i)=reshape(CELL.INCL(i).DS.DATA,size(CELL.POS,2),1);
% % end
% % LS=sort(LS,2);
%LS1=CELL.LSG1(:);
%LS2=CELL.LSG2(:);
LS1=reshape(CELL.DNK.DATA(1,:,:,:),[size(CELL.POS,2) 1]); 
LS2=reshape(CELL.DNK.DATA(2,:,:,:),[size(CELL.POS,2) 1]); 
PRE=CELL.PRE'+1;  
DIM=CELL.DIM; 
EQ=Equation;
%MAT=OPT(2);

% GRID
P=zeros(PRE);

% DISPLAY
% -------------------------------------------------------------------------
disp(' ');
disp('=======================================================');
disp('Third phase generation generation');
disp('-------------------------------------------------------');
disp(['|EQUATION/METHOD    : ' num2str(EQ)]);
disp('-------------------------------------------------------');

% OPERATING FUNCTION
% =========================================================================

% DEFAULT PARAMETERS
a = 0.002;      % EQ = 1, Coating term
b = 1;          % EQ = 1,3, LS2 exponent
c = 0.005;      % EQ = 2, Coating term
d = 1;          % EQ = 2, LS2 factor 
e = 0.045;      % EQ = 2, Bridge lenght
f = 0.2;        % EQ = 3, Fabric fraction
g = 0.2;        % EQ = 4, Fabric fraction
h = 0.5;        % EQ = 4, Bridge ration

TOL = 0.00001;   % Tolerance for iterative procedure

% READ INPUT PARAMETERS
if nargin==3
     if EQ==1
         a = Parameters(1);
         b = Parameters(2);
     elseif EQ==2
         c = Parameters(1);
         d = Parameters(2);
         e = Parameters(3);
     elseif EQ==3
         b = Parameters(1);
         f = Parameters(2);
     elseif EQ==4
         g = Parameters(1);
         h = Parameters(2);

     elseif EQ==5
         a = Parameters(1);
         amp = Parameters(2);
         freq = Parameters(3);
     else
         error('PROBLEM WITH TYPE OF EQUATION');
     end
end
 
% LS1 LS2 PRODUCT
% -------------------------------------------------------------------------
if EQ==1
    RVE = a-(LS1.*(LS2.^b)); 
    Total_Solid_Frac = sum(RVE >= 0) / numel(RVE); 

    
% COATING + BRIDGE
% -------------------------------------------------------------------------
elseif EQ==2
    RVE = -min(LS1-c,LS1+LS2*d-e);
    Total_Solid_Frac = sum(RVE >= 0) / numel(RVE);
    disp(['|FRACTION OF Total_Solid_Frac (inclusion_binding)        : ' num2str(Total_Solid_Frac)]);
% LS1 LS2 PRODUCT : FRACTION IMPOSED
% -------------------------------------------------------------------------
elseif EQ==3
    
    % PEPARATION
    Void1 = (LS1.*LS2.^b); % Base operating function 
    VINCL=sum(LS1<0)/numel(LS1); % Inclusion volume    
    Ai=min(Void1); % Initial minimal limit
    Af=max(Void1); % Initial maximal limit     
    OK=1; n=0; % Iteration master
    
    disp('ITERATIVE SOLVING FOR EQ 3...');
    
    % ITERATIVE SOLVING
    while OK  
        n=n+1;
        Ac=(Ai+Af)/2;                          % New trial        
        W=1-sum((Ac-Void1)<0)/numel(LS1)-VINCL-f; % New residual        
        if W>0 Af=Ac; else Ai=Ac; end          % Actualize limits 
        fprintf('Iter %3d : RESIDUS = %.6g\n', n, abs(W));
        if abs(W)<TOL OK=0; end              % STOP
    end
    
    % PROCEED 
    RVE=Void1 - Ac
    Total_Solid_Frac = sum(RVE >= 0) / numel(RVE);
    disp('-------------------------------------------------------');
    
% COATING + BRIDGE : FRACTION IMPOSED
% -------------------------------------------------------------------------
elseif EQ==4

    % PREPARATION
    BF = g*h; % Bridge fraction
    Void1 = LS1+LS2; % Base operating function 
    VINCL=sum(LS1<0)/numel(LS1); % Inclusion volume    
    Ai=min(Void1); % Initial minimal limit
    Af=max(Void1); % Initial maximal limit     
    OK=1; n=0; % Iteration master
    
    disp('ITERATIVE BRIDGE FORMING (EQ 4)...');    
    % ITERATIVE BRIDGE FORMING
    while OK  
        n=n+1;
        Ac=(Ai+Af)/2;% New trial        
        W=1-sum(-min(LS1,Void1-Ac)<0)/numel(LS1)-VINCL-BF; % New residual        
        if W>0 Af=Ac; else Ai=Ac; end % Actualize limits 
        fprintf('%3d : RESIDUS = %.6g\n', n, abs(W));
        if abs(W)<TOL OK=0; end  % STOP
    end
    
    % PROCEED
    Void=-min(LS1,Void1-Ac);
    ABRID=1-VINCL-sum(Void<0)/numel(LS1);
    e=Ac;
    
    disp('-------------------------------------------------------');
    
    disp('ITERATIVE COATING EXPANSION (EQ 4)...');    
    % ITERATIVE COATING EXPANSION
    Void1 = Void1-e; % Base operating function  
    Ai=0; % Initial minimal limit
    Af=max(LS1); % Initial maximal limit     
    OK=1; n=0; % Iteration master
    while OK  
        n=n+1;
        Ac=(Ai+Af)/2;% New trial        
        W=1-sum(-min(LS1-Ac,Void1)<0)/numel(LS1)-VINCL-g; % New residual        
        if W>0 Af=Ac; else Ai=Ac; end % Actualize limits 
        fprintf('%3d : RESIDUS = %.6g\n', n, abs(W));
        if abs(W)<TOL OK=0; end  % STOP
    end
    
    % PROCEED
    %Void=-min(LS1-Ac,Void1);
    RVE=min(LS1-Ac,Void1);
    Total_Solid_Frac = sum(RVE < 0) / numel(RVE);
    
    disp('-------------------------------------------------------');
    disp(['DEBUG CHECK:']);
    disp(['Analytical Volume (Target): ', num2str(CELL.VOL_INCL)]);
    disp(['Voxel Grid Volume (Actual): ', num2str(VINCL)]);
    disp(['Error: ', num2str(VINCL - CELL.VOL_INCL)]);
    disp('================================================');
    
elseif EQ==5
% =======================================================
    % ONE-SIDED BULGE / BUMP
    % =======================================================
    % Parameters(1) = Base Coating Thickness (can be 0)
    % Parameters(2) = Bulge Height (How big the bump is)
    % Parameters(3) = Bulge Direction (1=+X, 2=+Y, 3=+Z, 4=-X, etc.)
    
    base_thick = Parameters(1);
    bulge_h    = Parameters(2);
    axis_dir   = Parameters(3);
    
    % 1. Get the Coordinates relative to the inclusion center?
    % Since we don't have local coordinates easily, we use the Gradient 
    % (Surface Normal) to find "The Right Side" of the sphere.
    
    RVE_grid = reshape(LS1, PRE);
    [Gx, Gy, Gz] = gradient(RVE_grid);
    
    % 2. Select the side for the bump
    if axis_dir == 1,     G = Gx;  % East (+X)
    elseif axis_dir == 2, G = Gy;  % North (+Y)
    elseif axis_dir == 3, G = Gz;  % Top (+Z)
    elseif axis_dir == 4, G = -Gx; % West (-X)
    end
    
    % 3. Create the Mask
    % Only positive values mean we are on that specific side.
    % We use a power (e.g., ^4) to make the bump localized and round.
    
    Bump_Mask = reshape(max(0, G), size(LS1));
    Bump_Mask = Bump_Mask ./ (max(Bump_Mask(:)) + 1e-9); % Normalize 0-1
    
    % The Shape of the Bump: A smooth hill
    Localized_Bulge = bulge_h .* (Bump_Mask .^ 4); 
    
    % 4. Apply to Sphere
    % Total Thickness = Uniform Base + The Bump
    RVE = (base_thick + Localized_Bulge) - LS1;
    
    % 5. Standardize Solid = Positive
    % (Already done: (Thick - LS1) is positive inside coating)
    RVE=-RVE

    Total_Solid_Frac = sum(RVE < 0) / numel(RVE);
    
    disp('-------------------------------------------------------');
    disp(['| Single-Sided Bump Generated']);

end

Binding_frac = Total_Solid_Frac - VINCL;
Porosity_Frac = 1.0 - Total_Solid_Frac;
CELL.VINCL = VINCL;
DSI = new_grid(CELL.PRE(:), CELL.LIM(:));
DSI.DATA(:) = reshape(RVE, PRE); 
CAP = true;                                
SURF = iso_surf_ox(DSI, 0.0, CAP);
RVE = DSI.DATA;
CELL.RVE = RVE;

%% WORK CONTOURING
%% =========================================================================
%    CONT = struct;
%    X=reshape(CELL.POS(2,:),PRE);
%    Y=reshape(CELL.POS(1,:),PRE);
%    Z=reshape(CELL.POS(3,:),PRE);
%    [CONT.FACE CONT.VERT]=isosurface(X,Y,Z,reshape(RVE,PRE),0); % Contour

%% Void ADDITION TO CELL
%% =========================================================================

%% CREATE CELL
%NINCL=size(CELL.INCL,2)+1;
%CELL.INCL(NINCL).CONT=CONT;
%CELL.INCL(NINCL).DS.DATA=reshape(Void,PRE);
%CELL.INCL(NINCL).VOL=V;
%CELL.VOL_VOID=V;
%% DISPLAY
%% -------------------------------------------------------------------------

% --- DISPLAY ---
disp(['|FRACTION OF BINDING PHASE (Coating)  : ' num2str(Binding_frac)]);
disp(['|FRACTION OF INCLUSIONS IN RVE        : ' num2str(CELL.VINCL)]);
disp(['|FRACTION OF POROSITY (MATRIX)        : ' num2str(Porosity_Frac)]);
if EQ==4
    disp(['|BRIDGE FRACTION IN RVE              : ' num2str(ABRID)]);
%    disp(['|BRIDGE FRACTION IN BINDING PHASE    : ' num2str(ABRID/(1-NCELL.GOP(5)))]);
end
disp('=======================================================');
