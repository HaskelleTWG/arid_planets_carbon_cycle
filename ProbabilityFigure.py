
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import os

#   Directory containing CSV files  
csv_folder = "ProbabilityFigure"

#   Get all CSV files in the folder  
csv_files = glob.glob(f"{csv_folder}/*.csv")


#  custom labels: you will have to make changes to the labels for each plot. Additional labels are commented out.
file_labels = {
    #"Figures1_Earth_Escape_on_MC_Area_LowAlb_v4.csv": "Earth, Surface-dep., α = 0.3, Escape Included", #-dependent H$_2$O Ingassing", #α = 0.3",
    #"Figures1_Earth_Escape_on_MC_Volume_LowAlb_v6.csv": "Earth, Mass-dep., α = 0.3, Escape Included", #-dependent H$_2$O Ingassing", #α = 0.3",
    "Figures1_Earth_Escape_on_MC_Area_LowAlb_v4.csv": "Earth, Surface-dep., α = 0.3, Decreasing Melt",
    "Figures1_Earth_Escape_on_MC_Volume_LowAlb_v6.csv": "Earth, Mass-dep., α = 0.3",
    "Figures1_Earth_Escape_on_MC_Area_HighAlb_v2.csv": "Earth, Surface-dep., α =  0.65",
    "Figures1_Earth_Escape_on_MC_Volume_HighAlb_v2.csv": "Earth, Mass-dep., α =  0.65",
    "Figures1_Earth_Escape_on_MC_Area_HighMelt_v2.csv": r"Earth, Surface-dep., α = 0.3, $40\, \mathrm{km}^3/\mathrm{yr}$", #40
    "Figures1_Earth_Escape_on_MC_Area_ModernMelt_v2.csv": r"Earth, Surface-dep., α = 0.3, $20\, \mathrm{km}^3/\mathrm{yr}$", #20
    "Figures1_Earth_Escape_on_MC_None_v2.csv": "No Ingassing", #α = 0.3, H$_2$O "
    
    "Figures1_Earth_Escape_off_MC_Area_LowAlb_v2.csv": "Surface-dep.",#, α = 0.3, Escape Excluded",
    "Figures1_Earth_Escape_off_MC_Volume_LowAlb_v2.csv": "Mass-dep.", #α = 0.3, Escape Excluded",
    "Figures1_Earth_Escape_off_MC_None_v2.csv": "No Ingassing", #α = 0.3",

    "Figures1_Venus_Escape_off_MC_Area_HighAlb_v3.csv": "Venus, Surface-dep., α = 0.65, Escape Excluded",
    "Figures1_Venus_Escape_off_MC_Volume_HighAlb_v3.csv": "Venus, Mass-dep., α = 0.65, Escape Excluded",
    
    "Figures1_Venus_Escape_on_MC_Area_EarthHyp_v3.csv": "Venus, Surface-dep,, α = 0.65, Earth Hyp.",
    "Figures1_Venus_Escape_on_MC_Volume_EarthHyp_v3.csv": "Venus, Mass-dep., α = 0.65, Earth Hyp.",
    "Figures1_Venus_Escape_on_MC_Area_HighAlb_v2.csv": "Venus, Surface-dep., α = 0.65, Decreasing Melt",
    "Figures1_Venus_Escape_on_MC_Volume_HighAlb_v2.csv": "Venus, Mass-dep., α = 0.65", #Escape Included",
    "Figures1_Venus_Escape_on_MC_Area_HighMelt_v2.csv": r"Venus, Surface-dep., α = 0.65, $40\, \mathrm{km}^3/\mathrm{yr}$",
    "Figures1_Venus_Escape_on_MC_Area_ModernMelt_v2.csv": r"Venus, Surface-dep., α = 0.65, $20\, \mathrm{km}^3/\mathrm{yr}$",

    "Figures1_Earth_Escape_on_Max_Interior_H2O_tests_MC_Area_v5.csv": "Surface-dep., Crustal Hydration Only",
    "Figures1_Earth_Escape_on_Max_Interior_H2O_tests_MC_Volume_v5.csv": "Mass-dep., Crustal Hydration Only",
    "Figures1_Earth_Escape_off_Max_Interior_H2O_tests_MC_Area_v3.csv": "Surface-dep., Crustal Hydration Only",
    "Figures1_Earth_Escape_off_Max_Interior_H2O_tests_MC_Volume_v3.csv": "Mass-dep., Crustal Hydration Only"
    

}

###### Deep water cycle endmembers, Figure 6
fig_3 = "n"
if fig_3 == "y":
     # define keywords to filter files  
    exclude_keywords = ["Hyp", "Venus", "HighAlb", "Melt", "Escape_off", "99", "Interior"]  # skip files with these words
    include_keywords = ["LowAlb", "None"] # must include one of these
    
    # define custom order  
    def sort_key(filename):
        name = os.path.basename(filename)
        if "LowAlb" in name:
            planet = 0
        elif "Interior" in name:
            planet = 1
        elif "None" in name:
            planet = 1
        else:
            planet = 2

        # Determine melt scenario order if present: 40 km3/yr, then 20 km3/yr
        if "HighMelt" in name:
            melt_order = 0
        elif "ModernMelt" in name:
            melt_order = 1
        else:
            melt_order = 2  # fallback

        return (planet, melt_order)




    # filter files based on keywords  
    filtered_files = [
        f for f in csv_files 
        if any(kw in f for kw in include_keywords) and not any(kw in f for kw in exclude_keywords)

    ]

    # sort the filtered files using the custom key
    filtered_files = sorted(filtered_files, key=sort_key)

    # count files for color scaling  
    earth_files = [f for f in filtered_files if "Earth" in f and "LowAlb" in f]
    hyp_files = [f for f in filtered_files if "Earth" in f and "Interior" in f]
    venus_files = [f for f in filtered_files if "Earth" in f and "None" in f]

    # color maps  
    earth_colors = plt.cm.Blues(np.linspace(0.35, 0.8, len(earth_files)))  
    venus_colors = plt.cm.YlOrBr(np.linspace(0.3, 0.31, len(venus_files)))
    hyp_colors = plt.cm.RdPu(np.linspace(0.25, 0.45, len(hyp_files)))



    # plotting
    plt.figure(figsize=(8, 6))

    earth_idx, hyp_idx, venus_idx = 0, 0, 0  # track color index

    for csv_file in filtered_files:
        df = pd.read_csv(csv_file)
        
        # get custom label (or use filename if not provided)
        label = file_labels.get(os.path.basename(csv_file), os.path.basename(csv_file))

            # set color and linestyle based on parameters 
        if "Earth" in csv_file and "LowAlb" in csv_file:
            color = earth_colors[earth_idx]
            earth_idx += 1
            linestyle = '--' if "None" in csv_file else '-'
        elif "Earth" in csv_file and "Interior" in csv_file:
            color = hyp_colors[hyp_idx]
            hyp_idx += 1
            linestyle = '--' if "None" in csv_file else '-' 
        elif "Earth" in csv_file and "None" in csv_file:
            color = venus_colors[venus_idx]
            venus_idx += 1
            linestyle = '--' if "None" in csv_file else '-'


        else:
            color = "black"  # Fallback
            linestyle = '-'
  
        plt.plot(df["Initial_Water_Mass"], df["Probability"], 
                 marker='o', linestyle= linestyle, color=color, label=label)


    plt.xscale('log')
    plt.xlabel('Initial Surface Water Mass (Earth oceans)')
    plt.ylabel('Probability (Final Temperature > 400 K)')
    plt.legend(loc = "lower left")
    plt.grid(True)
    plt.xticks([0.001, 0.01, 0.1, 1.0], ["0.1%", "1%", "10%", "100%"])
    
    # save plot  
    name = "Earth_v7_Probability_Plot"
    save_path = os.path.join(csv_folder, name + ".png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight') 
    plt.show()






###### Albedo, Figure 7
fig_3 = "n"
if fig_3 == "y":
     #  define keywords to filter files  
    exclude_keywords = ["Hyp",  "Melt", "Escape_off", "99", "Interior", "None"]  # Skip files with these words
    include_keywords = ["LowAlb","HighAlb"] # Must include one of these
    
    #   define custom order  
    def sort_key(filename):
        name = os.path.basename(filename)
        # list Earth outputs, then Venus outputs in legend
        if "Earth" in name:
            planet = 0
        elif "Venus" in name:
            planet = 1
        elif "None" in name:
            planet = 1
        else:
            planet = 2

        # then sort by water ingassing type (area = surface-dependent, volume = mass-dependent)
        if "Area" in name:
            melt_order = 0
        elif "Volume" in name:
            melt_order = 1
        else:
            melt_order = 2  # fallback

        return (planet, melt_order)


    # filter files based on keywords  
    filtered_files = [
        f for f in csv_files 
        if any(kw in f for kw in include_keywords) and not any(kw in f for kw in exclude_keywords)

    ]

    # sort the filtered files using the custom key
    filtered_files = sorted(filtered_files, key=sort_key)

    # count files for color scaling  
    venus_files = [f for f in filtered_files if "Venus" in f and "HighAlb" in f]
    earth_files = [f for f in filtered_files if "Earth" in f] 

    # color maps  
    earth_colors = plt.cm.Blues(np.linspace(0.35, 0.85, len(earth_files)))  
    venus_colors = plt.cm.Oranges(np.linspace(0.3, 0.7, len(venus_files)))

    # plotting
    plt.figure(figsize=(8, 6))

    earth_idx, hyp_idx, venus_idx = 0, 0, 0  # track color index

    for csv_file in filtered_files:
        df = pd.read_csv(csv_file)
        
        # get custom label (or use filename if not defined)
        label = file_labels.get(os.path.basename(csv_file), os.path.basename(csv_file))

            # set color and linestyle based on parameters  
        if "Earth" in csv_file:
            color = earth_colors[earth_idx]
            earth_idx += 1
            linestyle = '--' if "HighAlb" in csv_file else '-'
        elif "Venus" in csv_file:
            color = venus_colors[venus_idx]
            venus_idx += 1
            linestyle = '--' if "HighAlb" in csv_file else '-' 
        elif "None" in csv_file and "None" in csv_file:
            color = hyp_colors[venus_idx]
            hyp_idx += 1
            linestyle = '--' if "None" in csv_file else '-'


        else:
            color = "black"  # Fallback
            linestyle = '-'

        plt.plot(df["Initial_Water_Mass"], df["Probability"], 
                 marker='o', linestyle= linestyle, color=color, label=label)


    plt.xscale('log')
    plt.xlabel('Initial Surface Water Mass (Earth oceans)')
    plt.ylabel('Probability (Final Temperature > 400 K)')
    plt.legend(loc = "lower left")
    plt.grid(True)
    plt.xticks([0.001, 0.01, 0.1, 1.0], ["0.1%", "1%", "10%", "100%"])
  
    name = "Albedo_v6_Probability_Plot"
    save_path = os.path.join(csv_folder, name + ".png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')  
    plt.show()




   
###### Escape, Figure 8 and S9
fig_3 = "n"
if fig_3 == "y":
     # define keywords to filter files  
    exclude_keywords = ["Hyp",  "Melt",  "99", "Interior", "None", "Volume"]  # skip files with these words
    include_keywords = ["LowAlb","HighAlb", "Escape_off","Escape_on", "Area"] # must include one of these
    
    #  define custom order  
    def sort_key(filename):
        name = os.path.basename(filename)
        # prioritize Earth outputs first, then Venus outputs
        if "Earth" in name:
            planet = 0
        elif "Venus" in name:
            planet = 1
        elif "None" in name:
            planet = 1
        else:
            planet = 2

        # then sort by escape 
        if "Escape_off" in name:
            melt_order = 0
        elif "Escape_on" in name:
            melt_order = 1
        else:
            melt_order = 2  # fallback

        return (planet, melt_order)



    filtered_files = [
        f for f in csv_files 
        if any(kw in f for kw in include_keywords) and not any(kw in f for kw in exclude_keywords)
        and "Figures1_Earth_Escape_on_MC_Area_HighAlb_v2.csv" not in f # remove a specific file
        and "Figures1_Earth_Escape_on_MC_Volume_HighAlb_v2.csv" not in f

    ]
    # sort the filtered files using the custom key
    filtered_files = sorted(filtered_files, key=sort_key)

    # count files for color scaling  
    venus_files = [f for f in filtered_files if "Venus" in f and "HighAlb" in f]
    earth_files = [f for f in filtered_files if "Earth" in f] 

    # color maps  
    earth_colors = plt.cm.Blues(np.linspace(0.35, 0.85, len(earth_files)))  
    venus_colors = plt.cm.Oranges(np.linspace(0.3, 0.7, len(venus_files)))

    # plotting
    plt.figure(figsize=(8, 6))

    earth_idx, hyp_idx, venus_idx = 0, 0, 0  # track color index

    for csv_file in filtered_files:
        df = pd.read_csv(csv_file)
        
        # get custom label (or use filename if not defined)
        label = file_labels.get(os.path.basename(csv_file), os.path.basename(csv_file))

        if "Earth" in csv_file:
            color = earth_colors[earth_idx]
            earth_idx += 1
            linestyle = '--' if "Escape_off" in csv_file else '-'
        elif "Venus" in csv_file:
            color = venus_colors[venus_idx]
            venus_idx += 1
            linestyle = '--' if "Escape_off" in csv_file else '-' 
        elif "None" in csv_file and "None" in csv_file:
            color = hyp_colors[venus_idx]
            hyp_idx += 1
            linestyle = '--' if "None" in csv_file else '-'


        else:
            color = "black"  # Fallback
            linestyle = '-'

        plt.plot(df["Initial_Water_Mass"], df["Probability"], 
                 marker='o', linestyle= linestyle, color=color, label=label)

    plt.xscale('log')
    plt.xlabel('Initial Surface Water Mass (Earth oceans)')
    plt.ylabel('Probability (Final Temperature > 400 K)')
    plt.legend(loc = "lower left")
    plt.grid(True)
    plt.xticks([0.001, 0.01, 0.1, 1.0], ["0.1%", "1%", "10%", "100%"])

    name = "Escape_v7_Probability_Plot"
    save_path = os.path.join(csv_folder, name + ".png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()




# effects of no water ingassing, Figure S4 and S5
crustal_hydration = "n"
if crustal_hydration == "y":
    #  define keywords to filter files  
    exclude_keywords = ["Hyp", "Venus", "HighAlb", "Melt", "Escape_off", "99"]  # skip files with these words
    include_keywords = ["Interior", "LowAlb", "None"] # must include one of these
    
    #  define custom order  
    def sort_key(filename):
        name = os.path.basename(filename)
        if "LowAlb" in name:
            planet = 0
        elif "Interior" in name:
            planet = 1
        elif "None" in name:
            planet = 1
        else:
            planet = 2

        # determine melt scenario order if present: 40 km3/yr, then 20 km3/yr
        if "HighMelt" in name:
            melt_order = 0
        elif "ModernMelt" in name:
            melt_order = 1
        else:
            melt_order = 2  # fallback

        return (planet, melt_order)


    #  filter files based on keywords  
    filtered_files = [
        f for f in csv_files 
        if any(kw in f for kw in include_keywords) and not any(kw in f for kw in exclude_keywords)
        and "Figures1_Earth_Escape_on_MC_Area_LowAlb_v3.csv" not in f
        and "Figures1_Earth_Escape_on_MC_Volume_LowAlb_v5.csv" not in f

    ]

    # sort the filtered files using the custom key
    filtered_files = sorted(filtered_files, key=sort_key)


    # count files for color scaling  
    earth_files = [f for f in filtered_files if "Earth" in f and "LowAlb" in f]
    hyp_files = [f for f in filtered_files if "Earth" in f and "Interior" in f]
    venus_files = [f for f in filtered_files if "Earth" in f and "None" in f]

    # color maps  
    earth_colors = plt.cm.Blues(np.linspace(0.35, 0.8, len(earth_files)))  
    venus_colors = plt.cm.YlOrBr(np.linspace(0.3, 0.31, len(venus_files)))
    hyp_colors = plt.cm.RdPu(np.linspace(0.25, 0.45, len(hyp_files)))


    # plotting
    plt.figure(figsize=(8, 6))

    earth_idx, hyp_idx, venus_idx = 0, 0, 0  # track color index

    for csv_file in filtered_files:
        df = pd.read_csv(csv_file)
        
        # Get custom label (or use filename if not defined)
        label = file_labels.get(os.path.basename(csv_file), os.path.basename(csv_file))
 
        if "Earth" in csv_file and "LowAlb" in csv_file:
            color = earth_colors[earth_idx]
            earth_idx += 1
            linestyle = '--' if "None" in csv_file else '-'
        elif "Earth" in csv_file and "Interior" in csv_file:
            color = hyp_colors[hyp_idx]
            hyp_idx += 1
            linestyle = '--' if "None" in csv_file else '-'  
        elif "Earth" in csv_file and "None" in csv_file:
            color = venus_colors[venus_idx]
            venus_idx += 1
            linestyle = '--' if "None" in csv_file else '-'


        else:
            color = "black"  # Fallback
            linestyle = '-'

        plt.plot(df["Initial_Water_Mass"], df["Probability"], 
                 marker='o', linestyle= linestyle, color=color, label=label)

    plt.xscale('log')
    plt.xlabel('Initial Surface Water Mass (Earth oceans)')
    plt.ylabel('Probability (Final Temperature > 400 K)')
    plt.legend(loc = "lower left")
    plt.grid(True)
    plt.xticks([0.001, 0.01, 0.1, 1.0], ["0.1%", "1%", "10%", "100%"])
  
    name = "MaxInterior_v6_Probability_Plot"
    save_path = os.path.join(csv_folder, name + ".png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight') 
    plt.show()



### Sensitivity to planetary hypsometry, Figure S8
hypsometry_plot = "n"
if hypsometry_plot == "y":
    # define keywords to filter files  
    exclude_keywords = ["Melt", "Escape_off", "99", "Interior"]  # skip files with these words
    include_keywords = ["LowAlb", "Hyp", "HighAlb"] # must include one of these
    
    #  define custom order  
    def sort_key(filename):
        name = os.path.basename(filename)
        if "LowAlb" in name:
            planet = 0
        elif "HighAlb" in name:
            planet = 1
        elif "Hyp" in name:
            planet = 2
        else:
            planet = 3

        # determine melt scenario order if present: 40 km3/yr, then 20 km3/yr
        if "HighMelt" in name:
            melt_order = 0
        elif "ModernMelt" in name:
            melt_order = 1
        else:
            melt_order = 2  # fallback

        return (planet, melt_order)




    # filter files based on keywords  
    filtered_files = [
        f for f in csv_files 
        if any(kw in f for kw in include_keywords) and not any(kw in f for kw in exclude_keywords)
        and "Figures1_Earth_Escape_on_MC_Area_HighAlb_v2.csv" not in f
        and "Figures1_Earth_Escape_on_MC_Volume_HighAlb_v2.csv" not in f

    ]

    # sort the filtered files using the custom key
    filtered_files = sorted(filtered_files, key=sort_key)

    # count files for color scaling  
    earth_files = [f for f in filtered_files if "Earth" in f and "LowAlb" in f]
    venus_files = [f for f in filtered_files if "Venus" in f and "HighAlb" in f]
    hyp_files = [f for f in filtered_files if "Venus" in f and "Hyp" in f]

    # color maps  
    earth_colors = plt.cm.Blues(np.linspace(0.35, 0.8, len(earth_files)))  
    hyp_colors = plt.cm.Greys(np.linspace(0.35, 0.8, len(hyp_files)))
    venus_colors = plt.cm.Oranges(np.linspace(0.3, 0.7, len(venus_files)))

    # plotting 
    plt.figure(figsize=(8, 6))

    earth_idx, hyp_idx, venus_idx = 0, 0, 0  # track color index

    for csv_file in filtered_files:
        df = pd.read_csv(csv_file)
        
        # Get custom label (or use filename if not defined)
        label = file_labels.get(os.path.basename(csv_file), os.path.basename(csv_file))

        if "Earth" in csv_file and "LowAlb" in csv_file:
            color = earth_colors[earth_idx]
            earth_idx += 1
            linestyle = '--' if "None" in csv_file else '-'
        elif "Venus" in csv_file and "Hyp" in csv_file:
            color = hyp_colors[hyp_idx]
            hyp_idx += 1
            linestyle = '--' if "Hyp" in csv_file else '-'  
        elif "Venus" in csv_file and "HighAlb" in csv_file:
            color = venus_colors[venus_idx]
            venus_idx += 1
            linestyle = '--' if "None" in csv_file else '-'


        else:
            color = "black"  # Fallback
            linestyle = '-'

        #  plotting 
        plt.plot(df["Initial_Water_Mass"], df["Probability"], 
                 marker='o', linestyle= linestyle, color=color, label=label)

 
    plt.xscale('log')
    plt.xlabel('Initial Surface Water Mass (Earth oceans)')
    plt.ylabel('Probability (Final Temperature > 400 K)')
    plt.legend(loc = "lower left")
    plt.grid(True)
    plt.xticks([0.001, 0.01, 0.1, 1.0], ["0.1%", "1%", "10%", "100%"])

    name = "Hyp_v6_Probability_Plot"
    save_path = os.path.join(csv_folder, name + ".png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()







## Melt production, Figure S10
melt_plot = "y"
if melt_plot == "y":
    #  define keywords to filter files  
    exclude_keywords = ["Hyp", "Escape_off", "99", "Interior", "Volume"]  # skip files with these words
    include_keywords = ["LowAlb", "Melt", "HighAlb"] # must include one of these
    
    # define custom order  
    def sort_key(filename):
        name = os.path.basename(filename)

        if "Earth" in name:
            planet = 0
        elif "Venus" in name:
            planet = 1
        elif "Hyp" in name:
            planet = 2
        else:
            planet = 3

        # determine melt scenario order: 40 km3/yr, then 20 km3/yr
        if "HighMelt" in name:
            melt_order = 0
        elif "ModernMelt" in name:
            melt_order = 1
        else:
            melt_order = 2  # fallback

        return (planet, melt_order)


    # filter files based on keywords  
    filtered_files = [
        f for f in csv_files 
        if any(kw in f for kw in include_keywords) and not any(kw in f for kw in exclude_keywords)
        and "Figures1_Earth_Escape_on_MC_Area_HighAlb_v2.csv" not in f
        and "Figures1_Earth_Escape_on_MC_Volume_HighAlb_v2.csv" not in f

    ]

    # sort the filtered files using the custom key
    filtered_files = sorted(filtered_files, key=sort_key)


    #  cunt files for color scaling  
    earth_files = [f for f in filtered_files if "Earth" in f]
    venus_files = [f for f in filtered_files if "Venus" in f]

    # color maps 
    earth_colors = plt.cm.Blues(np.linspace(0.35, 0.8, len(earth_files)))  
    venus_colors = plt.cm.Oranges(np.linspace(0.3, 0.7, len(venus_files)))

    # plotting 
    plt.figure(figsize=(8, 6))

    earth_idx, hyp_idx, venus_idx = 0, 0, 0  # track color index

    for csv_file in filtered_files:
        df = pd.read_csv(csv_file)
        
        # Get custom label (or use filename if not defined)
        label = file_labels.get(os.path.basename(csv_file), os.path.basename(csv_file))
  
        if "Earth" in csv_file:
            color = earth_colors[earth_idx]
            earth_idx += 1
            linestyle = '--' if "HighMelt" in csv_file else '-'
        elif "Venus" in csv_file:
            color = venus_colors[venus_idx]
            venus_idx += 1
            linestyle = '--' if "HighMelt" in csv_file else '-'  
        else:
            color = "black"  # Fallback
            linestyle = '-'

        # plotting 
        plt.plot(df["Initial_Water_Mass"], df["Probability"], 
                 marker='o', linestyle= linestyle, color=color, label=label)

    plt.xscale('log')
    plt.xlabel('Initial Surface Water Mass (Earth oceans)')
    plt.ylabel('Probability (Final Temperature > 400 K)')
    plt.legend(loc = "lower left")
    plt.grid(True)
    plt.xticks([0.001, 0.01, 0.1, 1.0], ["0.1%", "1%", "10%", "100%"])

    name = "Melt_v6_Probability_Plot"
    save_path = os.path.join(csv_folder, name + ".png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
   
