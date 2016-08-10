commandwindow;

sz = [68, 156];
fover = 

B = [16 16];

G = ceil(sz./B);

I = zeros(sz);

%r = rem(sz(1), B(1));
r = G(1)*B(1) - sz(1); % residual
B(2) = floor(sz(1)/ r);
r2 = B(2)*r;

for bx = 0:G(1)-1
	for by = 0:1%0:G(2)-1
		for tx = 0:B(1)-1
			for ty = 0:B(2)-1
				i = tx + bx*B(1) + r*ty + (r2-sz(1))*by;
				j = ty + by*(B(2)+ 1);
				if(i < 0) continue;
				end
				if(i >= sz(1) )
					i = i - sz(1);
					j = j + 1;
				end
				
				if(0 && i >= sz(1) )
					i = i - sz(1);
					j = j + 1;
				end
				
				if(i >= sz(1) )
					%continue
				end
				if(j < sz(2) )
					I(i+1,j+1) = 3*bx + ty+10;
					%l = sub2ind(sz, i+1, j+1);
					%if mod(l-1,16) ==0
					%	I(i+1,j+1) = 20;
					%end
				end
			end
		end
	end
end

figure(1); clf;
imagesc(I); colormap jet;