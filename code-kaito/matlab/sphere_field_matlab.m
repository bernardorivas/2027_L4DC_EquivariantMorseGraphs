function sphere_field_matlab(paperPalette)
% MATLAB-native sphere field: exact ODE, standard gem palette, no training.
if nargin<1,paperPalette=false;end
root=fileparts(fileparts(mfilename('fullpath')));
out=fullfile(root,'artifacts','sphere-matlab');if ~exist(out,'dir'),mkdir(out);end
C=orderedcolors('gem');
if paperPalette
 out=fullfile(root,'artifacts','sphere-matlab-paper');if ~exist(out,'dir'),mkdir(out);end
 C(1,:)=[.4 .4 .4];C(5,:)=[0 114 178]/255;C(4,:)=[204 121 167]/255;C(2,:)=[0 158 115]/255;C(3,:)=[230 159 0]/255;
end
el=18*pi/180;
d=[cos(el),0,sin(el)];u=[0,1,0];w=cross(d,u);
f=figure('Visible','off','Color','w','Position',[100 100 800 760]);
ax=axes(f,'Position',[.03 .03 .94 .94]);hold(ax,'on');
[x,y,z]=sphere(100);
surf(ax,x,y,z,'FaceColor',[.88 .93 .98],'FaceAlpha',.23, ...
 'EdgeColor','none','FaceLighting','gouraud','AmbientStrength',.7, ...
 'DiffuseStrength',.4,'SpecularStrength',.15);
% All visible glyphs are evaluated in 3-D before rendering.
p=[];
for j=0:7
 b=-.80+j*.225;
 for a=-.8+mod(j,2)*.11:.225:.85
  if a*a+b*b>.87^2,continue;end
  q=a*u+b*w+sqrt(1-a*a-b*b)*d;
  if min(abs(q(3)-[-.4,0,.4]))<.04,continue;end
  p(end+1,:)=q; %#ok<AGROW>
 end
end
v=field(p);assert(max(abs(sum(p.*v,2)))<1e-12);
v=.14*v./vecnorm(v,2,2);base=1.008*p;
quiver3(ax,base(:,1),base(:,2),base(:,3),v(:,1),v(:,2),v(:,3),0, ...
 'Color',C(1,:),'LineWidth',1.15,'MaxHeadSize',.7);
phi=linspace(0,2*pi,1001)';
for k=1:3
 h=[-.4,0,.4];height=h(k);r=sqrt(1-height^2);
 P=[r*cos(phi),r*sin(phi),height+zeros(size(phi))];
 colors=C([5,4,2],:);color=colors(k,:);
 front=P*d'>=0;
 for side=[false,true]
  Q=P;Q(front~=side,:)=NaN;
  col=color;if ~side,col=.72+.28*color;end
  ls='-';if k==2,ls='--';end
  plot3(ax,Q(:,1),Q(:,2),Q(:,3),'Color',col,'LineWidth',1.8,'LineStyle',ls);
 end
 labels={'$A_-$','$C$','$A_+$'};
 target=1.18*u+height*cos(el)*w;
 text(ax,target(1),target(2),target(3),labels{k},'Interpreter','latex', ...
   'FontSize',20,'Color',color,'HorizontalAlignment','left');
end
for h=[-1,1]
 color=C(3,:);if h<0
 if paperPalette,color=[213 94 0]/255;end
 color=.65+.35*color;end
 plot3(ax,0,0,h,'o','Color',color,'MarkerFaceColor','w','MarkerSize',6,'LineWidth',1.2);
 P=1.11*h*w;
 label='$N$';if h<0,label='$S$';end
 text(ax,P(1),P(2),P(3),label,'Interpreter','latex','FontSize',20, ...
 'Color',color,'HorizontalAlignment','center');
end
% Orthographic silhouette; no reference latitude/longitude mesh.
P=cos(phi)*u+sin(phi)*w;
plot3(ax,P(:,1),P(:,2),P(:,3),'Color',[.62 .67 .72],'LineWidth',.7);
axis(ax,'equal');axis(ax,'off');axis(ax,[-1.25 1.25 -1.25 1.45 -1.25 1.25]);
campos(ax,6*d);camtarget(ax,[0,0,0]);camup(ax,[0,0,1]);camproj(ax,'orthographic');
light(ax,'Position',[-1,-3,5],'Style','infinite');
exportgraphics(ax,fullfile(out,'sphere-field.png'),'Resolution',400,'BackgroundColor','white');
if ~paperPalette
exportgraphics(ax,fullfile(out,'sphere-field.pdf'),'ContentType','vector','BackgroundColor','white');
end
savefig(f,fullfile(out,'sphere-field.fig'));
close(f);
fid=fopen(fullfile(out,'README.md'),'w');
fprintf(fid,['# MATLAB sphere field\n\nGenerated with MATLAB R2026b using surf, quiver3 and orderedcolors("gem").\n' ...
 'The actual ODE is evaluated at front-surface points; arrows are normalized\n' ...
 'in 3D to show direction rather than speed. The sphere is translucent;\n' ...
 'rear invariant circles and the south-pole marker are faint.\n' ...
 'This palette is an alternative to the Bernardo palette in the paper bundle.\n' ...
 'Run sphere_field_matlab from code-kaito/matlab. The FIG is editable in MATLAB.\n']);fclose(fid);
disp(out)
end
function V=field(P)
z=P(:,3);V=.6*[-P(:,2),P(:,1),zeros(size(z))]+z.*(.16-z.^2).*([0,0,1]-z.*P);
end
